import matplotlib.pyplot as plt 
import pandas as pd
import argparse 
import os
import seaborn as sns
sns.set()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-e', '--environment', default = 'PongNoFrameskip-v4',
                        type = str, help = 'ALE name of environment')
    parser.add_argument('-xl', '--xlabel', default = 'iteration',
                        type = str, help = 'label of x axis')
    parser.add_argument('-yl', '--ylabel', default = 'score',
                        type = str, help = 'label of y axis')
    parser.add_argument('-xc', '--xcolumn', default = 'iteration',
                        type = str, help = 'column of x axis')
    parser.add_argument('-yc', '--ycolumn', default = 'avg_score',
                        type = str, help = 'column of y axis')
    parser.add_argument('-t', '--title', default = '',
                        type = str, help = 'plot title')
    parser.add_argument('-n', '--name', default = 'plot',
                        type = str, help = 'plot name')
    
    args = parser.parse_args()
    path = 'results/' + args.environment + '/'
    
    directories = os.listdir(path)
    
    for directory in directories:   
        data = pd.read_csv(path + directory + '/logs_score.txt', sep = ',')
        plt.plot(data[args.xcolumn], data[args.ycolumn], label = directory)
            
    plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.legend(loc='upper left')
    
    plt.savefig(args.name + ".png")
    
    
    
    
    
    