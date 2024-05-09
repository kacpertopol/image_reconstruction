#!/usr/bin/env python3

import torch
import os
import argparse
import sys
import configparser
from configparser import ConfigParser
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import shutil

def make_generations(maxg , f = 1.5708 , dtype = torch.float32 , device = "cpu"):
    """
    Creates and initializes parameters for approximating the probability distribution
    p of Beta emission.

    Arguments:

    maxg - int, number of generations
    f - initial value for the parameters, (1 + cos(f = 1.5708)) / 2 is  approximately 0.5 - the initial fraction
    dtype - floating point type, gpus handle float32 I think
    device - device where the tensors are allocated

    Returns:

    generationsx , generationsy - parameters for the x and y sections
    p - probability in rectangles
    dtype - floating point type, gpus handle float32 I think
    device - device where the tensors are allocated
    maxg - number of generations

    """
    generationsx = [
                torch.full([pow(2 , g) , pow(2 , g)] , f , dtype = dtype , device = device , requires_grad = True)
                for g in range(maxg)
            ]
    generationsy = [
                torch.full([pow(2 , g) , pow(2 , g)] , f , dtype = dtype , device = device , requires_grad = True)
                for g in range(maxg)
            ]

    p = torch.ones([pow(2 , maxg) , pow(2 , maxg)] , dtype = dtype , device = device , requires_grad = True)

    return {"generationsx" : generationsx , "generationsy" : generationsy , "dtype" : dtype , "device" : device , "maxg" : maxg , "p" : p}

def gen_to_list(gen):
    """
    Takes the parameters for approximating the probability distribution
    p of Beta emission (created eg. in make_generations)
    and returns a list for an optimizer (eg. torch.optim.Adam).

    Arguments:

    gen - parameters for approximating the probability distribution of Beta
          emission p, created eg. in make_generations

    Returns:

    List of tensors for use in eg. torch.optim.Adam
    """
    l = []
    for m in gen["generationsx"]:
        l.append(m)
    for m in gen["generationsy"]:
        l.append(m)
    l.append(gen["p"])
    return l

def get_info(gen , margin = 0.2):
    """
    Takes the parameters for approximating the probability distribution
    p of Beta emission (created eg. in make_generations) and
    returns four matrices that contain the x , y thicknesses
    of the rectangles, the x , y position of the lower left corners
    of the rectangles and the probabilities in the rectangles.

    Arguments:

    gen - parameters for approximating the probability distribution of Beta
          emission p, created eg. in make_generations

    Returns:

    thicknessx - matrix containing the x thickness of the rectangles
    thicknessy - matrix containing the y thickness of the rectangles
    minx - matrix containing the x posisiton of the lower left corners of the rectanlges
    miny - matrix containing the y posisiont of the lower left corners of the rectangles
    p - matrix containing the probability in the rectangles
    """
    t_mul_x = torch.tensor([[1.0 , -1.0] , [1.0 , -1.0]] , dtype = gen["dtype"] , device = gen["device"])
    t_add_x = torch.tensor([[0.0 , 1.0] , [0.0 , 1.0]] , dtype = gen["dtype"], device = gen["device"])
    
    t_mul_y = torch.tensor([[1.0 , 1.0] , [-1.0 , -1.0]] , dtype = gen["dtype"] , device = gen["device"])
    t_add_y = torch.tensor([[0.0 , 0.0] , [1.0 , 1.0]] , dtype = gen["dtype"], device = gen["device"])
    

    tx = torch.tensor([[1.0]] , dtype = gen["dtype"], device = gen["device"])
    ty = torch.tensor([[1.0]] , dtype = gen["dtype"], device = gen["device"])
    lx = torch.tensor([[0.0]] , dtype = gen["dtype"], device = gen["device"])
    ly = torch.tensor([[0.0]] , dtype = gen["dtype"], device = gen["device"])
    for g in range(gen["maxg"]):
        fx = margin + (1.0 - 2.0 * margin) * 0.5 * (torch.cos(gen["generationsx"][g]) + 1.0)
        fy = margin + (1.0 - 2.0 * margin) * 0.5 * (torch.cos(gen["generationsy"][g]) + 1.0)
       
        lx = torch.kron(lx , torch.ones([2 , 2] , dtype = gen["dtype"], device = gen["device"])) \
                + torch.kron(tx , torch.ones([2 , 2] , dtype = gen["dtype"], device = gen["device"])) * \
                  t_add_x.repeat(fx.shape) * \
                  torch.kron(fx , torch.ones([2 , 2], device = gen["device"]))
        ly = torch.kron(ly , torch.ones([2 , 2] , dtype = gen["dtype"], device = gen["device"])) \
                + torch.kron(ty , torch.ones([2 , 2] , dtype = gen["dtype"], device = gen["device"])) * \
                  t_add_y.repeat(fy.shape) * \
                  torch.kron(fy , torch.ones([2 , 2], device = gen["device"]))
        
        tx = torch.kron(tx , torch.ones([2 , 2] , dtype = gen["dtype"], device = gen["device"])) * \
                (t_add_x.repeat(fx.shape) +  torch.kron(fx , t_mul_x))
        ty = torch.kron(ty , torch.ones([2 , 2] , dtype = gen["dtype"], device = gen["device"])) * \
                (t_add_y.repeat(fy.shape) +  torch.kron(fy , t_mul_y))

    prob = 0.5 * (torch.cos(gen["p"]) + 1.0)
    prob = prob / prob.sum()
    #print("gen[\"p\"] : " , gen["p"])
    #print("prob : " , prob)
 
    parameters = 0
    for g in gen["generationsx"]:
        parameters += g.shape[0] * g.shape[1]
    for g in gen["generationsy"]:
        parameters += g.shape[0] * g.shape[1]
    parameters += gen["p"].shape[0] * gen["p"].shape[1]
    parameters = torch.tensor([[parameters]])
    
    return {"thicknessx" : tx , "thicknessy" : ty , "minx" : lx , "miny" : ly , "prob" : prob , "parameters" : parameters}

def get_p1(info , data , sigma = 0.1 , batch = None):
    """
    Returns 
    -log(likelyhood) 
    of observing data.

    Arguments:
    info - eg. result of get_info
    data - [[x1 , y1] , [x2 , y2] , ...] the data as a torch tensor
    sigma - we assume that the probability distribution of gamma-gamma
            relative to beta emission is a Gaussian with standard deviation sigma
    """
    mx = info["minx"]
    my = info["miny"]
    tx = info["thicknessx"]
    ty = info["thicknessy"]
    prob = info["prob"]

    if(batch is None):
        x = data[: , 0]
        y = data[: , 1]

        px = torch.erf((mx - x[: , None , None]) / (np.sqrt(2.0) * sigma)) 
        px = px - torch.erf((mx + tx - x[: , None , None]) / (np.sqrt(2.0) * sigma))
        
        py = torch.erf((my - y[: , None , None]) / (np.sqrt(2.0) * sigma)) 
        py = py - torch.erf((my + ty - y[: , None , None]) / (np.sqrt(2.0) * sigma))

        p = torch.sum(0.25 * px * py * prob / (tx * ty) , (1 , 2))

        return torch.sum(-torch.log(p)) / data.shape[0]
    else:

        start = 0
        result = None

        while start < data.shape[0]:
            x = data[start : start + batch , 0]
            y = data[start : start + batch , 1]

            px = torch.erf((mx - x[: , None , None]) / (np.sqrt(2.0) * sigma)) 
            px = px - torch.erf((mx + tx - x[: , None , None]) / (np.sqrt(2.0) * sigma))
            
            py = torch.erf((my - y[: , None , None]) / (np.sqrt(2.0) * sigma)) 
            py = py - torch.erf((my + ty - y[: , None , None]) / (np.sqrt(2.0) * sigma))

            p = torch.sum(0.25 * px * py * prob / (tx * ty) , (1 , 2))

            if(start == 0):
                result = torch.sum(-torch.log(p)) / data.shape[0]
            else:
                result += torch.sum(-torch.log(p)) / data.shape[0]
            
            start += batch

        return result

def write(path , dct , num = None):
    """
    Writes matrices from dictionary to files.

    Arguments:

    path - path to directory, files will end up in directory
    dct - a dictionary with matrices
    num - additional number that will be added to the file names
    """
    extra = ""
    if(num is not None):
        extra = "_" + str(num)
    for key in dct.keys():
        with open(os.path.join(path , key + extra) , "w") as f:
            for row in dct[key]:
                for col in row:
                    f.write(str(col.item()) + " ")
                f.write("\n")

def write_data(path , data):
    """
    Writes the data.

    Arguments:

    path - path to directory, a new file data will be created in this directory
    data - [[x1 , y1] , [x2 , y2] , ...]
    """
    with open(os.path.join(path , "data") , "w") as f:
        for row in data:
            f.write(str(row[0]) + " " + str(row[1]) + "\n")

def read_data(path ,  dtype = torch.float32 , device = "cpu" , fullpath = False):
    """
    Writes the data.

    Arguments:

    path - path to directory, a new file data will be created in this directory
    data - [[x1 , y1] , [x2 , y2] , ...]
    """
    pth = os.path.join(path , "data")
    if(fullpath):
        pth = path
    dta = []
    with open(pth , "r") as f:
        for row in f.readlines():
            x , y = row.strip().split()
            dta.append([float(x) , float(y)])
    
    d = torch.tensor(dta , dtype = dtype , device = device)

    return d[torch.randperm(d.size()[0])]

def write_density(path , info , num = None , nxy = 100):

    x = info["minx"]
    y = info["miny"]
    tx = info["thicknessx"]
    ty = info["thicknessy"]

    prob = info["prob"]

    xx = torch.zeros((nxy , nxy) , dtype = x.dtype , device = x.device)
    yy = torch.zeros((nxy , nxy) , dtype = x.dtype , device = x.device)
    ttxx = torch.zeros((nxy , nxy) , dtype = x.dtype , device = x.device)
    ttyy = torch.zeros((nxy , nxy) , dtype = x.dtype , device = x.device)

    for ixx in range(nxy):
        for iyy in range(nxy):
            xx[ixx , iyy] = ixx / nxy
            yy[ixx , iyy] = iyy / nxy
            ttxx[ixx , iyy] = 1.0 / nxy
            ttyy[ixx , iyy] = 1.0 / nxy

    interx = torch.maximum(xx[: , : , None , None] , x)
    intery = torch.maximum(yy[: , : , None , None] , y)
    intertx = torch.minimum(xx[: , : , None , None]  + ttxx[: , : , None , None], x + tx) - interx
    interty = torch.minimum(yy[: , : , None , None]  + ttyy[: , : , None , None], y + ty) - intery

    intertx = torch.where(intertx < 0.0 , 0.0 , intertx)
    interty = torch.where(interty < 0.0 , 0.0 , interty)

    d = torch.sum(nxy * nxy * intertx * interty * prob / (tx * ty) , (2 , 3))

#            for r in range(x.shape[0]):
#                for c in range(x.shape[1]):
#                    xv = x[r , c]
#                    yv = y[r , c]
#                    txv = tx[r , c]
#                    tyv = ty[r , c]
#
#                    interx = max(xv , xxv) 
#                    intery = max(yv , yyv) 
#                    intertx = min(xv + txv , xxv + ttxxv) - interx
#                    interty = min(yv + tyv , yyv + ttyyv) - intery
#
#                    if(intertx >= 0 and interty >= 0):
#                        d[ixx , iyy] += nxy * nxy * intertx * interty * prob[r , c] / (txv * tyv) 

    name = "density"
    if(num is not None):
        name += "_" + str(num)
    with open(os.path.join(path , name) , "w") as f:
        for iy in reversed(range(d.shape[0])):
            for ix in range(d.shape[1]):
                f.write(str(d[ix , iy].item()) + " ")
            f.write("\n")

#if(__name__ == "__main__"):
#    parser = argparse.ArgumentParser(prog = "Test section.")
#    parser.add_argument("path" , help = "Path for result, this schould be an existing directory containing data.")
#    parser.add_argument("generations" , type = int , help = "Number of generations.")
#    parser.add_argument("--batch" , "-b" , type = int , default = 1000 , help = "Batch size.")
#    parser.add_argument("--scale" , "-s" , type = float , default = 0.05 , help = "Scale for normal distribution.")
#    parser.add_argument("--iterations" , "-i" , type = int , default = 1000 , help = "Number of iterations.")
#    parser.add_argument("--learningrate" , "-l" , type = float , default = 0.01 , help = "Learning rate.")
#    parser.add_argument("--margin" , "-m" , type = float , default = 0.2 , help = "Section fraction will not be smaller then the margin.")
#    parser.add_argument("--device" , "-d" , help = "Device for computatios, cpu or cuda.")
#    args = parser.parse_args()
#
#    dev = "cpu"
#    if(args.device == "cuda" and torch.cuda.is_available()):
#        dev = "cuda"
#
#    data = read_data(args.path , device = dev)
#
#    # generations for test
#    gen = make_generations(args.generations , device = dev)
#
#    # parameters
#    parameters = gen_to_list(gen)
#
#    # optimizer
#    optimizer = torch.optim.Adam(parameters , lr = args.learningrate)
#    #optimizer = torch.optim.ASGD(parameters , lr = args.learningrate)
#
#    #
#    info = get_info(gen)
#    write(args.path , info)
#    
#    #
#    print("starting training loop for " , info["parameters"][0,0].item() , "parameters")
#    for i in range(args.iterations):
#
#        loss = None
#        ib = 0
#        start = 0
#        while start + args.batch - 1 <= data.shape[0]:
#            sys.stdout.write("batch " + str(ib) + "/" + str(int(data.shape[0] / args.batch)) + " " * 10)
#
#            info = get_info(gen , margin = args.margin)
#
#            dta = data[start : start + args.batch]
#
#            l = get_p1(info , dta , sigma = args.scale)
#            loss = -l.item()
#
#            optimizer.zero_grad()
#            l.backward()
#            optimizer.step()
#
#            sys.stdout.write("\r")
#            ib += 1
#            start += args.batch
#
#        sys.stdout.write("\n")
#
#        print("in iteration " , i , "/" , args.iterations , " logp : " , loss)
#        
#        info = get_info(gen , margin = args.margin)
#        write(args.path , info , i)
#        
#        with open(os.path.join(args.path , "log") , "a") as f:
#            f.write(str(-l.item()) + "\n")

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(prog = "Test section.")
    parser.add_argument("config" , help = "Path to configuration file.")
    parser.add_argument("output" , help = "Path to output directory.")
    args = parser.parse_args()

    # READ CONFIGURATION FILE

    config = ConfigParser()
    config.read(args.config)

    args_path = config["data"].get("path" , fallback = configparser._UNSET)

    args_batch = config["likelyhood"].getint("batch" , fallback = configparser._UNSET)
    args_iterations = config["likelyhood"].getint("iterations" , fallback = configparser._UNSET) + 1
    args_learningrate = config["likelyhood"].getfloat("learningrate" , fallback = configparser._UNSET)
    args_writel = config["likelyhood"].getint("writeevery" , fallback = configparser._UNSET)
    args_valid = config["likelyhood"].get("valid" , fallback = configparser._UNSET)

    args_citerations = config["metropolishastings"].getint("citerations" , fallback = configparser._UNSET)
    args_reject = config["metropolishastings"].getint("reject" , fallback = configparser._UNSET)
    args_write = config["metropolishastings"].getint("writeevery" , fallback = configparser._UNSET)
    args_dw = config["metropolishastings"].getfloat("dw" , fallback = configparser._UNSET)
    args_df = config["metropolishastings"].getfloat("df" , fallback = configparser._UNSET)
    args_mcb = config["metropolishastings"].getint("batch" , fallback = configparser._UNSET)

    args_scale = config["global"].getfloat("scale" , fallback = configparser._UNSET)
    args_device = config["global"].get("device" , fallback = configparser._UNSET).strip()
    args_maxp = config["global"].getint("maxp" , fallback = configparser._UNSET)
    args_nxy = config["global"].getint("nxy" , fallback = configparser._UNSET)

    print("- MAKING OUTPUT DIRECTORY -")

    if(not os.path.isdir(args.output)):
        os.mkdir(args.output)
    else:
        sys.stderr.write("Output directory exists. Exiting.")
        sys.exit(1)
    args_dir = args.output
 
    # COPY CONFIGURATION FILE AND DATA TO OUTPUT DIRECTORY
    
    print("- COPY CONFIGURATION FILE AND DATA TO OUTPUT DIRECTORY -")

    shutil.copyfile(args_path , os.path.join(args_dir , "data"))
    config["data"]["path"] = os.path.abspath(os.path.join(args_dir , "data"))
    
    with open(os.path.join(args_dir , "config") , "w") as f:
        config.write(f)

    # CPU OR CUDA

    dev = "cpu"
    if(args_device == "cuda" and torch.cuda.is_available()):
        dev = "cuda"
        print("- USING CUDA -")
    else:
        print("- USING CPU -")

    # READ DATA

    print("- READ DATA -")

    data = read_data(args_path , device = dev , fullpath = True)

    valid = read_data(args_valid , device = dev , fullpath = True)
    
    print("... valid.shape[0] : " , valid.shape[0])

    print("... data.shape[0] : " , data.shape[0])

    # GENERATE INITIAL CONFIGURATION FOR SECTIONS

    print("- GENERATE INITIAL CONFIGURATION FOR SECTIONS -")
    
    gen = make_generations(args_maxp , device = dev)

    # PARAMETERS FOR OPTIMIZER

    print("- PARAMETERS FOR OPTIMIZER -")
    
    parameters = gen_to_list(gen)

    # OPTIMIZER

    print("- OPTIMIZER -")
    
    optimizer = torch.optim.Adam(parameters , lr = args_learningrate)

    # DATA TO CALCULATE LOG P

    print("- DATA TO CALCULATE LOG P -")
    
    info = get_info(gen)

    write(args_dir , info)
    write_density(args_dir , info , nxy = args_nxy)
    
    # TRAINING LOOP

    print("- TRAINING LOOP -")

    if(args_iterations - 1 > 0 and args_batch <= data.shape[0]):
        print("... starting training loop using " , args_maxp * args_maxp , "pixels")

        i = 0 # index for writting
        start = 0 # start of batch
        alliter = 0 # total number of iterations
        while alliter < args_iterations:
            print("... start , start + args_batch , data.shape[0]: " , start , start + args_batch , data.shape[0])
            
            dta = data[start : start + args_batch]

            info = get_info(gen)
            l = get_p1(info , dta , sigma = args_scale)
            loss = -l.item()
 
            if(alliter % args_writel == 0):                
                print("... writting info")
                with torch.no_grad():
                        
                    write(args_dir , info , num = i)

                    write_density(args_dir , info , num = i , nxy = args_nxy)

                    losstrainfull = -get_p1(info , data , sigma = args_scale , batch = args_batch).item()
                    lossvalidfull = -get_p1(info , valid , sigma = args_scale , batch = args_batch).item()
                    
                    with open(os.path.join(args_dir , "log") , "a") as f:
                        f.write(str(loss) + " " + str(lossvalidfull) + " " + str(losstrainfull) + "\n")
                    
                    print("... in iteration " , alliter , "/" , args_iterations - 1 , ", logp train , logp valid full , logp train full: " , loss , lossvalidfull , losstrainfull)

                    i += 1
 
            if(alliter < args_iterations - 1): 
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

            start += args_batch
            if start + args_batch > data.shape[0]:
                start = 0
            
            alliter += 1

    # MC LOOP

    print("-MC LOOP-")

    t = args_reject
    if(args_citerations + t > 0):
        dw = args_dw
        df = args_df
        with torch.no_grad():
            info = get_info(gen)

            dta = data
            logp = -get_p1(info , dta , sigma = args_scale , batch = args_mcb) * data.shape[0]
            print("Starting mc mh loop for " , args_maxp * args_maxp , "points")
            if(not os.path.isdir(os.path.join(args_dir , "mc"))):
                os.mkdir(os.path.join(args_dir , "mc"))

            idensity = 0
            accepted = 0
            alliter = 0
            while alliter < args_citerations + t:
                dta = data
                
                wchange = torch.rand((pow(2 , args_maxp) , pow(2 , args_maxp)) , device = args_device) * 2 * dw - dw

                gen["p"] += wchange

                fxchange = []
                ddf = df
                for gx in gen["generationsx"]:
                    fxchange.append(torch.rand(gx.shape , device = args_device) * 2 * ddf - ddf)
                    ddf *= 2.0

                for ix in range(len(gen["generationsx"])):
                    gen["generationsx"][ix] += fxchange[ix]

                fychange = []
                ddf = df
                for gy in gen["generationsy"]:
                    fychange.append(torch.rand(gy.shape , device = args_device) * 2 * ddf - ddf)
                    ddf *= 2.0

                for iy in range(len(gen["generationsy"])):
                    gen["generationsy"][iy] += fychange[iy]
                
                newinfo = get_info(gen)
                newlogp = -get_p1(newinfo , dta , sigma = args_scale , batch = args_mcb) * data.shape[0]

                currentAccepted = 0
                acc = min(1.0 , torch.exp(newlogp - logp).item())
                print("... in iteration " + str(alliter + 1) + "/" + str(args_citerations + t) + " acc , logp / ... , newlogp / ... , fraction accepted: " , 
                        acc , logp.item() / data.shape[0] , newlogp.item() / data.shape[0] , float(accepted) / (alliter + 1))
                if uniform.rvs(loc = 0.0 , scale = 1.0) < acc:
                    logp = newlogp
                    accepted += 1
                    currentAccepted = 1
                else:
                    gen["p"] -= wchange
                    for ix in range(len(gen["generationsx"])):
                        gen["generationsx"][ix] -= fxchange[ix]
                    for iy in range(len(gen["generationsy"])):
                        gen["generationsy"][iy] -= fychange[iy]
                with open(os.path.join(args_dir , "mc" , "log") , "a") as f:
                    f.write(str(acc) + " " + str(logp.item() / data.shape[0]) + " " + str(newlogp.item() / data.shape[0]) + " " + str(currentAccepted) + "\n")
                if(alliter >= t and alliter % args_write == 0):                
                    write_density(os.path.join(args_dir , "mc") , newinfo , num = idensity , nxy = args_nxy)
                    idensity += 1
                alliter += 1
            print("... accepted : " , 100.0 * float(accepted) / (args_citerations + t) , "%")

           
