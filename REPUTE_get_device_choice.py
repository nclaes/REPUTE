import numpy as np
import pyopencl as cl
from pyopencl import array
import sys
import subprocess
import time
import os
import argparse

def main():
	all_platforms = dict()
	platforms = cl.get_platforms()
	num_plts = 0
	for pltm in platforms:
		if(pltm.name[:12] != 'Experimental'):	
			if(pltm.get_devices(cl.device_type.CPU) != []):
				all_platforms[num_plts] = ["CPU", pltm.get_devices(cl.device_type.CPU), cl.Context(devices=pltm.get_devices(cl.device_type.CPU)),pltm]
				num_plts = num_plts + 1

	for pltm in platforms:
		if(pltm.name[:12] != 'Experimental'):	
			if(pltm.get_devices(cl.device_type.GPU) != []):
				all_platforms[num_plts] = ["GPU",pltm.get_devices(cl.device_type.GPU), cl.Context(devices=pltm.get_devices(cl.device_type.GPU)),pltm]
				num_plts = num_plts + 1
	num_devices = 0
	all_devices = dict()
	for key,value in all_platforms.items():
		for i in value[1]:
			all_devices[num_devices] = [value[0],value[2], i, value[3]] ## Device type, Context, Device name
			num_devices = num_devices + 1
	print('----------------------------------------------------------------------')
	print('Following is the list of platforms:')
	print('{:<15}'.format('PLATFORM NO.'),':','{:<5}'.format('TYPE'),':','PLATFORM NAME/S\n')
	for key,value in all_platforms.items():
		print('{:^15}'.format(key),':','{:^5}'.format(value[0]),':',value[1],'\n')

	print('----------------------------------------------------------------------\n')
	print('Following is the list of all devices:')
	print('{:<15}'.format('DEVICE NO.'),':','{:<5}'.format('TYPE'),':','DEVICE NAME\n')
	for key, value in all_devices.items():
		print('{:^15}'.format(key),':','{:<5}'.format(value[0]),':',value[2],'\n')
	print('----------------------------------------------------------------------\n')


if __name__ == '__main__':
	main()
