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

def make_points(maxp , dtype = torch.float32 , device = "cpu"):
    """
    Creates and initializes parameters for approximating the probability distribution
    p of Beta emission. 

    Arguments:

    maxp - int, number of Gaussians
    dtype - floating point type, gpus handle float32 I think
    device - device where the tensors are allocated

    Returns:

    x , y - coordinates of the centers of the Gaussian distributions
    sigma - standard deviation of gaussians
    pw - weights for each gaussian
    dtype - floating point type, gpus handle float32 I think
    device - device where the tensors are allocated
    maxp - number of Gaussians

    """
    # center of gaussian in x
    x = torch.linspace(0.0 , 1.0 , maxp , dtype = dtype , device = device).repeat(maxp)
    x.requires_grad_()
    # center of gaussian in y
    y = torch.linspace(0.0 , 1.0 , maxp , dtype = dtype , device = device).repeat_interleave(maxp)
    y.requires_grad_()
    # standard deviaion of gaussian
    sigma = torch.ones(maxp * maxp , dtype = dtype , device = device , requires_grad = True)
    # weight of gaussian
    pw = torch.ones(maxp * maxp , dtype = dtype , device = device , requires_grad = True) 

    return {"x" : x , "y" : y , "dtype" : dtype , "device" : device , "maxp" : maxp , "sigma" : sigma , "pw" : pw}

def points_to_list(gen , weight = False):
    """
    Takes the parameters for approximating the probability distribution
    p of Beta emission (created eg. in make_points)
    and returns a list for an optimizer (eg. torch.optim.Adam).

    Arguments:

    gen - parameters for approximating the probability distribution of Beta
          emission p, created eg. in make_points

    Returns:

    List of tensors for use in eg. torch.optim.Adam
    """
    l = []
    l.append(gen["x"])
    l.append(gen["y"])
    l.append(gen["sigma"])
    if(weight):
        print("... weights of Gaussians will be adjusted")
        l.append(gen["pw"])
    else:
        print("... weights of Gaussians will not be adjusted")
    return l

def get_info(gen , margin , weightmargin):
    """
    Takes the parameters for approximating the probability distribution
    p of Beta emission (created eg. in make_points) and
    returns data for calculating the log p. The distribution will be approximated using
    $$p(x , y | ...) = \frac{\sum_{i = 1}^{maxp} prob_{i} N(xcoord_{i} , ycoord_{i} , stdv_{i}^{2})}{...}$$

    Arguments:

    gen - parameters for approximating the probability distribution of Beta
          emission p, created eg. in make_points
    margin - the minimum standard deviation for Gaussians

    Returns:

    xcoord , ycoord - x and y coordinates of the centers of the Gaussians
    stdv - standard deviations of Gaussians
    prob - weights of Gaussians
    """
    px = gen["x"] # 0 ... 1
    py = gen["y"] # 0 ... 1
    psigma = margin + gen["sigma"] * gen["sigma"]
    #prob = 0.5 * (torch.cos(gen["pw"]) + 1.0) ## pw * pw ?
    prob = gen["pw"] * gen["pw"]
    prob = (weightmargin / prob.shape[0]) + (1.0 - weightmargin) * prob / prob.sum() 
    return {"xcoord" : px , "ycoord" : py , "stdv" : psigma , "prob" : prob}

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
    px = info["xcoord"]
    py = info["ycoord"]
    psigma = info["stdv"] 
    prob = info["prob"]

    if(batch is None):

        x = data[: , 0]
        y = data[: , 1]

        p = torch.sum(
                prob * torch.exp(
                        -0.5 * 
                            ((x[: , None] - px) * (x[: , None] - px) + (y[: , None] - py) * (y[: , None] - py)) / 
                                (psigma * psigma + sigma * sigma)) / (2.0 * torch.pi * (psigma * psigma + sigma * sigma))
                , 1) 

        return torch.sum(-torch.log(p)) / data.shape[0]

    else:

        start = 0
        result = None

        while start < data.shape[0]:

            x = data[start : start + batch , 0]
            y = data[start : start + batch , 1]

            p = torch.sum(
                    prob * torch.exp(
                            -0.5 * 
                                ((x[: , None] - px) * (x[: , None] - px) + (y[: , None] - py) * (y[: , None] - py)) / 
                                    (psigma * psigma + sigma * sigma)) / (2.0 * torch.pi * (psigma * psigma + sigma * sigma))
                    , 1) 

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
        if(isinstance(dct[key] , torch.Tensor)):
            with open(os.path.join(path , key + extra) , "w") as f:
                for row in dct[key]:
                    f.write(str(row.item()) + " ")
                f.write("\n")

def write_density(path , info , num = None , nxy = 100 , device = "cpu"):
    px = info["xcoord"]
    py = info["ycoord"]
    psigma = info["stdv"] 
    prob = info["prob"]


    xl = torch.Tensor([i * (1.0 / nxy) + (0.5 / nxy) for i in range(nxy)]).to(device)
    yl = torch.Tensor([i * (1.0 / nxy) + (0.5 / nxy) for i in range(nxy)]).to(device)
    d = torch.zeros((xl.shape[0] , yl.shape[0]) , device = device)
    for ix in range(xl.shape[0]):
        for iy in range(yl.shape[0]):
            x = xl[ix : ix + 1]
            y = yl[iy : iy + 1]

            p = torch.sum(
                prob * torch.exp(
                    -0.5 * 
                        ((x[: , None] - px) * (x[: , None] - px) + (y[: , None] - py) * (y[: , None] - py)) / 
                            (psigma * psigma)) / (2.0 * torch.pi * (psigma * psigma))
                    , 1) 

            d[ix , iy] = p

    name = "density"
    if(num is not None):
        name += "_" + str(num)
    with open(os.path.join(path , name) , "w") as f:
        for iy in reversed(range(yl.shape[0])):
            for ix in range(xl.shape[0]):
                f.write(str(d[ix , iy].item()) + " ")
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
    args_dsigma = config["metropolishastings"].getfloat("dsigma" , fallback = configparser._UNSET)
    args_dxy = config["metropolishastings"].getfloat("dxy" , fallback = configparser._UNSET)
    args_dw = config["metropolishastings"].getfloat("dw" , fallback = configparser._UNSET)
    args_mcb = config["metropolishastings"].getint("batch" , fallback = configparser._UNSET)

    args_scale = config["global"].getfloat("scale" , fallback = configparser._UNSET)
    args_margin = config["global"].getfloat("margin" , fallback = configparser._UNSET)
    args_weightmargin = config["global"].getfloat("weightmargin" , fallback = configparser._UNSET)
    args_weight = config["global"].getboolean("weight" , fallback = configparser._UNSET)
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

    print("... data.shape[0] : " , data.shape[0])
    print("... valid.shape[0] : " , valid.shape[0])

    # GENERATE INITIAL PARAMETERS FOR GAUSSIANS

    print("- GENERATE INITIAL PARAMETERS FOR GAUSSIANS -")
    
    gen = make_points(args_maxp , device = dev)

    # PARAMETERS FOR OPTIMIZER

    print("- PARAMETERS FOR OPTIMIZER -")
    
    parameters = points_to_list(gen , weight = args_weight)

    # OPTIMIZER

    print("- OPTIMIZER -")
    
    optimizer = torch.optim.Adam(parameters , lr = args_learningrate)

    # DATA TO CALCULATE LOG P

    print("- DATA TO CALCULATE LOG P -")
    
    info = get_info(gen , args_margin , args_weightmargin)

    write(args_dir , info)
    write_density(args_dir , info , device = args_device , nxy = args_nxy)
    
    # TRAINING LOOP

    print("- TRAINING LOOP -")

    if(args_iterations - 1 > 0 and args_batch <= data.shape[0]):
        print("... starting training loop using " , args_maxp * args_maxp , "gaussians")

        i = 0 # index for writting
        start = 0 # start of batch
        alliter = 0 # total number of iterations
        while alliter < args_iterations:
            print("... start , start + args_batch , data.shape[0]: " , start , start + args_batch , data.shape[0])

            dta = data[start : start + args_batch]

            info = get_info(gen , args_margin , args_weightmargin)
            l = get_p1(info , dta , sigma = args_scale)
            loss = -l.item()

            if(alliter % args_writel == 0):                
                print("... writting info")
                with torch.no_grad():
                        
                    write(args_dir , info , num = i)
                    
                    write_density(args_dir , info , num = i , device = args_device , nxy = args_nxy)
                 
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
        dsigma = args_dsigma
        dxy = args_dxy
        dw = args_dw
        with torch.no_grad():
            info = get_info(gen , args_margin , args_weightmargin)

            dta = data
            logp = -get_p1(info , dta , sigma = args_scale , batch = args_mcb) * data.shape[0]
            print("... starting mc mh loop for " , args_maxp * args_maxp , "points")
            if(not os.path.isdir(os.path.join(args_dir , "mc"))):
                os.mkdir(os.path.join(args_dir , "mc"))

            idensity = 0
            accepted = 0
            alliter = 0
            while alliter < args_citerations + t:
                dta = data
                
                sigmachange = torch.rand(args_maxp * args_maxp , device = args_device) * 2 * dsigma - dsigma
                xchange = torch.rand(args_maxp * args_maxp , device = args_device) * 2 * dxy - dxy
                ychange = torch.rand(args_maxp * args_maxp , device = args_device) * 2 * dxy - dxy
                wchange = torch.rand(args_maxp * args_maxp , device = args_device) * 2 * dw - dw

                gen["sigma"] += sigmachange
                gen["x"] += xchange
                gen["y"] += ychange
                if(args_weight):
                    gen["pw"] += wchange
                
                newinfo = get_info(gen , args_margin , args_weightmargin)
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
                    gen["sigma"] -= sigmachange
                    gen["x"] -= xchange
                    gen["y"] -= ychange
                    if(args_weight):
                        gen["pw"] -= wchange
                with open(os.path.join(args_dir , "mc" , "log") , "a") as f:
                    f.write(str(acc) + " " + str(logp.item() / data.shape[0]) + " " + str(newlogp.item() / data.shape[0]) + " " + str(currentAccepted) + "\n")
                if(alliter >= t and alliter % args_write == 0):                
                    write_density(os.path.join(args_dir , "mc") , info , num = idensity , device = args_device , nxy = args_nxy)
                    idensity += 1
                alliter += 1
            print("... accepted : " , 100.0 * float(accepted) / (args_citerations + t) , "%")


