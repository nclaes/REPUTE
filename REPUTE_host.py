import numpy as np
import pyopencl as cl
from pyopencl import array
import sys
import subprocess
import time
import ast
import collections
from itertools import islice
import pickle
import math
import statistics
import os
import argparse

# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
def HCF(x, y):
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller+1):
        if((x % i == 0) and (y % i == 0)):
            hcf = i
            
    return hcf

def print_platforms_and_devices():
	all_platforms = dict()
	platforms = cl.get_platforms()
	num_plts = 0
	for pltm in platforms:
		if(pltm.name[:12] != 'Experimental'):	
			if(pltm.get_devices(cl.device_type.CPU) != []):
				all_platforms[num_plts] = ["CPU", pltm.get_devices(cl.device_type.CPU), cl.Context(devices=pltm.get_devices(cl.device_type.CPU)),pltm]
				num_plts = num_plts + 1
				# all_platforms.append(pltm.get_devices(cl.device_type.CPU))

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
	return all_devices

def opencl(genome, reads, n, e, device_choices, no_of_reads, no_of_outputs_per_read, tally, SA, F, total_memory_required, share_per_device,q_len):	
	no_of_reads = np.uint32(no_of_reads)
	# coverage = np.uint32(math.ceil((no_of_reads*n)/len(genome)))	
	device_list = print_platforms_and_devices()  ## Device type, Context, Device name	
	n = np.uint32(n)
	chosen_device_parameters = dict()
	sufficient_memory = 0	
	for key,value in device_list.items():
		if(value[2].global_mem_size > total_memory_required):
			sufficient_memory = 1
			# print(device_list[i].global_mem_size, total_memory_required, sufficient_memory)
		else:
			sufficient_memory = 0
			# print(device_list[i].global_mem_size, total_memory_required, sufficient_memory)		
		chosen_device_parameters[key] = [value[1], cl.CommandQueue(value[1], value[2], cl.command_queue_properties.PROFILING_ENABLE), sufficient_memory, value[3], value[2], value[0]]

	strand_of_read, mapped_endpos_read, cand_locs_per_read = launching_kernel(reads, genome, chosen_device_parameters, no_of_reads, n, no_of_outputs_per_read, e, tally, SA, F, share_per_device, device_choices,q_len, device_list)
	return strand_of_read, mapped_endpos_read, cand_locs_per_read



#  The function below i.e. cpu_opencl only has make_sugar kernel
def launching_kernel(reads, genome, chosen_device_parameters, no_of_reads, read_len, no_of_outputs_per_read, e, tally, SA, F, share_per_device, device_choices,q_len, device_list):
	
	program_file = open('REPUTE_kernel.cl', 'r')
	program_text = program_file.read()
	program_file.close()	
	no_of_outputs_per_read = np.uint32(no_of_outputs_per_read)
	word_size = 32
	q_gram_len = np.uint32(q_len)	
	print('{:<30}'.format('Genome length'),':', len(genome))
	print('{:<30}'.format('Total no of reads'),':',no_of_reads)
	print('{:<30}'.format('Minimum k-mer length'),':', q_gram_len)
	#-------------------------------------------------------------------------------------------------------
	#Setting kernel compilation parameters: passing constants to the kernel
	program_compilation_options = ["-Werror","-cl-mad-enable","-cl-denorms-are-zero"]#,"-cl-opt-disable" ,"-cl-std=CL1.2" ,"-cl-uniform-work-group-size"
	temp_text = "-DRLEN=" + str(read_len)
	program_compilation_options.append(temp_text)
	temp_text = "-DWORD_LENGTH=" + str(word_size)
	program_compilation_options.append(temp_text)
	temp_text = "-DPERMISSIBLE_ERROR=" + str(e)
	program_compilation_options.append(temp_text)
	temp_text = "-DCANDIDATES_PER_READ=" + str(no_of_outputs_per_read)
	program_compilation_options.append(temp_text)
	temp_text = "-DUINT_WITH_MSB_ONE=" + str(2147483648)
	program_compilation_options.append(temp_text)
	temp_text = "-DMIN_QGRAM_LEN=" + str(q_gram_len)  # minimum q-gram length - 1 (minus one), minus one just to prevent an extra computation
	program_compilation_options.append(temp_text)
	print('-----------------------------------------------------------------------')
	print('IGNORE THE FOLLOWING MESSAGE')
	num_reads_alloted = 0
	program = dict()
	kernel = dict()
	Genome = dict()
	Reads = dict()
	buffer_SA = dict()
	buffer_tally = dict()
	buffer_F = dict()
	cand_locs_per_read = dict()
	buffer_cand_locs_per_read = dict()
	strand_of_read = dict()
	buffer_strand_of_read = dict()
	mapped_endpos_read = dict()
	buffer_mapped_endpos_read = dict()
	num_work_items_per_device = dict()
	# print(device_choices)
	for i in device_choices:
		# index = device_choices.index(i)
		program[i] = cl.Program(chosen_device_parameters[i][0], program_text)
	# for i in program:
		try:
			program[i].build(program_compilation_options)  #
		except:
			print('Problem in building the kernel')
			# print("Build log:")
			# print(program[i].get_build_info(chosen_device_parameters[i][1],cl.program_build_info.LOG))
			raise
		print('-----------------------------------------------------------------------')
		kernel[i] = cl.Kernel(program[i], 'repute')
		if(chosen_device_parameters[i][5] == "CPU"):
			num_work_items_per_device[i] = int(HCF(share_per_device[i], 128))
		elif(chosen_device_parameters[i][5] == "GPU"):
			num_work_items_per_device[i] = int(HCF(share_per_device[i], 32))

		print('{:<25}'.format('Device selected'),':',chosen_device_parameters[i][4].name)
		print('{:<26}{:<2}{:.1f}'.format('Global MEM size (MB)',':',(chosen_device_parameters[i][4].global_mem_size)/(1024*1024)), 'MB')
		print('{:<25}'.format('Local MEM size (Bytes)'),':',(chosen_device_parameters[i][4].local_mem_size))
		print('{:<25}'.format('Max work group size'),':',(chosen_device_parameters[i][4].max_work_group_size))
		print('{:<25}'.format('Max work item size'),':',(chosen_device_parameters[i][4].max_work_item_sizes),'\n')
		print('{:<75}'.format('KERNEL USAGE INFORMATION'))
		print('{:<75}'.format('Kernel Name'),':',kernel[i].get_info(cl.kernel_info.FUNCTION_NAME))
		print('{:<75}'.format('Maximum Work group size'),':',kernel[i].get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE,chosen_device_parameters[i][4]))
		print('{:<75}'.format('A multiple for determining work-group sizes that ensure best performance'),':',kernel[i].get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,chosen_device_parameters[i][4]))
		print('{:<75}'.format('Local memory used by the kernel in bytes'),':',kernel[i].get_work_group_info(cl.kernel_work_group_info.LOCAL_MEM_SIZE,chosen_device_parameters[i][4]))
		print('{:<75}'.format('Private memory used by the kernel in bytes'),':',kernel[i].get_work_group_info(cl.kernel_work_group_info.PRIVATE_MEM_SIZE,chosen_device_parameters[i][4]))
		print('{:<75}'.format('Work items per work group'),':',num_work_items_per_device[i],'\n')

		if(chosen_device_parameters[i][5] == "CPU"):# That means its CPU, so the flags will be different
			# print('Do not copy data from host to device',i, chosen_device_parameters[i][2], chosen_device_parameters[i][3])
			Genome[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=genome)
			kernel[i].set_arg(0, Genome[i])
			Reads[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=reads[i])
			kernel[i].set_arg(1, Reads[i])
			buffer_SA[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=SA)
			kernel[i].set_arg(2, buffer_SA[i])
			buffer_tally[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=tally)
			kernel[i].set_arg(3, buffer_tally[i])
			buffer_F[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR , hostbuf=F)
			kernel[i].set_arg(4, buffer_F[i])
			cand_locs_per_read[i] = np.zeros(share_per_device[i], np.uint32)
			buffer_cand_locs_per_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=cand_locs_per_read[i])
			kernel[i].set_arg(5, buffer_cand_locs_per_read[i])
			strand_of_read[i] = np.zeros(no_of_outputs_per_read*share_per_device[i], np.uint8)
			buffer_strand_of_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=strand_of_read[i])
			kernel[i].set_arg(6, buffer_strand_of_read[i])
			mapped_endpos_read[i] = np.zeros(no_of_outputs_per_read*share_per_device[i], np.uint32)
			buffer_mapped_endpos_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=mapped_endpos_read[i])
			kernel[i].set_arg(7, buffer_mapped_endpos_read[i])			
		else:
			# print(chosen_device_parameters[i][5],'******************************')
			# print('Copy data from host to device',i, chosen_device_parameters[i][2], chosen_device_parameters[i][3])
			Genome[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=genome)
			kernel[i].set_arg(0, Genome[i])
			Reads[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=reads[i])
			kernel[i].set_arg(1, Reads[i])
			buffer_SA[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=SA)
			kernel[i].set_arg(2, buffer_SA[i])
			buffer_tally[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=tally)
			kernel[i].set_arg(3, buffer_tally[i])
			buffer_F[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR , hostbuf=F)
			kernel[i].set_arg(4, buffer_F[i])
			cand_locs_per_read[i] = np.zeros(share_per_device[i], np.uint32)
			buffer_cand_locs_per_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cand_locs_per_read[i])
			kernel[i].set_arg(5, buffer_cand_locs_per_read[i])
			strand_of_read[i] = np.zeros(no_of_outputs_per_read*share_per_device[i], np.uint8)
			buffer_strand_of_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=strand_of_read[i])
			kernel[i].set_arg(6, buffer_strand_of_read[i])
			mapped_endpos_read[i] = np.zeros(no_of_outputs_per_read*share_per_device[i], np.uint32)
			buffer_mapped_endpos_read[i] = cl.Buffer(chosen_device_parameters[i][0], cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mapped_endpos_read[i])
			kernel[i].set_arg(7, buffer_mapped_endpos_read[i])		
		num_reads_alloted += share_per_device[i]		

	event = dict()
	start_time = time.time()
	for i in device_choices:
		event[i] = cl.enqueue_nd_range_kernel(chosen_device_parameters[i][1], kernel[i], (share_per_device[i],) , (num_work_items_per_device[i],))
		# event[i] = cl.enqueue_task(chosen_device_parameters[i][1], kernel[i], wait_for=None)

	for i in device_choices:
		event[i].wait()
		if(chosen_device_parameters[i][5] != "CPU"):
			print('Transfer operation involved')
			q1 = cl.enqueue_copy(chosen_device_parameters[i][1], cand_locs_per_read[i], buffer_cand_locs_per_read[i])
			q2 = cl.enqueue_copy(chosen_device_parameters[i][1], strand_of_read[i], buffer_strand_of_read[i])
			q3 = cl.enqueue_copy(chosen_device_parameters[i][1], mapped_endpos_read[i], buffer_mapped_endpos_read[i])
			q1.wait(); q2.wait(); q3.wait()

	total_time = time.time() - start_time
	print('{:<25}{:<2}{:.4f}'.format('Mapping time',':',total_time),'s')
	#print(cand_locs_per_read[0],sum(cand_locs_per_read[0]), max(cand_locs_per_read[0]), min(cand_locs_per_read[0]), 'No of zeros =', len(cand_locs_per_read[0]) - np.count_nonzero(cand_locs_per_read[0]))
	#print('{:<25}{:<2}{:.4f}'.format('Mapping time',':',total_time),'s')
	return strand_of_read, mapped_endpos_read, cand_locs_per_read

def Write_Output(cand_locs_per_read, strand, mapped_endpos_read, device_choices, sequences, sequence_names, no_of_outputs_per_read, filename, n):
	heading = 'First column: Read name; 	Second column: Strand;		Third column: End position in the genome where read matched;	Fourth column: Edit distance; 		Fifth columbn: Read sequence\n'
	arr = []
	read_number = 0
	with open(filename+'.repute','w') as f:
		f.write(heading)
		for i in device_choices:
			temp_list = list(cand_locs_per_read[i])					
			for j in range(len(temp_list)):		
				for k in range(temp_list[j]):
					if(strand[i][j*no_of_outputs_per_read + k] >= 128):
						strnd = 'F'
						ed = strand[i][j*no_of_outputs_per_read + k] - 128
					else:
						strnd = 'R'
						ed = strand[i][j*no_of_outputs_per_read + k]
					if(k == 0):
						arr = [sequence_names[read_number][1:], strnd, str(mapped_endpos_read[i][j*no_of_outputs_per_read + k]), str(ed), str(sequences[read_number])]#, str(n)
					else:
						arr = [sequence_names[read_number][1:], strnd, str(mapped_endpos_read[i][j*no_of_outputs_per_read + k]), str(ed), "\""]#, str(n)
					# f.write("{:25}{:^10}{:^22}{:^8}{:^10}{:^150}".format(*arr)+'\n')
					f.write('\t'.join(arr[0:]) + '\n')
				read_number += 1

def readFastq(filename):
	sequence_names = []
	sequences = []
	qualities = []
	with open(filename) as fh:
		while True:
			name = fh.readline().rstrip()
			seq = fh.readline().rstrip()  # read base sequence
			fh.readline()  # skip placeholder line
			qual = fh.readline().rstrip() # base quality line
			if len(seq) == 0:
				break
			sequence_names.append(name)
			sequences.append(seq)
			qualities.append(qual)
	return sequences, qualities, sequence_names

def main():	
	print('-----------------------------------------------------------------------')
	np.set_printoptions(threshold=sys.maxsize)
	# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
	p = subprocess.run(['pwd'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip() + '/'

	parser = argparse.ArgumentParser(description="REPUTE : REad maPper  for  heterogeneoUs sysTEms . This tool maps short-reads to reference genome.")
	parser.add_argument("Fastq_filename", help="Give the name of reads file in fastq format.", type=str)
	parser.add_argument("Read_length", help="Provide the length of the reads. Choices: 100 or 150", type=int, default=100)
	parser.add_argument("Error", help="Provide the maximum permissible error. Smaller the value faster will be the algorithm. Range:[0-8]", type=int, default=5)
	parser.add_argument("Outputs", help="Maximum number of mappings allowed per read. Range:[1:3500]", type=int, default=100)
	parser.add_argument("klen", help="K-mer length. Range:[12:25]", type=int, default=12)
	parser.add_argument("-dc",help="Provide the numbers of all the devices to be used. These numbers can be from running 'get_device_choice.py' script", type=int, nargs='+')
	parser.add_argument("-rs",help="Provide the share of reads to be mapped for each device, with the sum accumulating to the total number of reads e.g. for a given 100000 should lead to a share of [50000 30000 20000] for three devices. All shares MUST be integers. Number of reads in the multiples of 2 will lead to better performance, especially for GPUs.", type=float, nargs='+')
	parser.add_argument("-nr",help="Number of reads to be mapped, default: all the reads in the fastq file", type=int)
	
	args = parser.parse_args()
	
	#---Read integer coded genome file 
	genome = np.load(p + 'genome_uppercase.npy', allow_pickle=False)
	#---Read suffix array
	SA =  np.load(p+'SA.npy', allow_pickle=False);
	#---Read Tally matrix of the FM Index
	tally = np.load(p+'Tally.npy', allow_pickle=False)
	#---Read F array of the FM Index giving cummulated count of the bases A,C,G and T
	F = np.load(p+'F_FMIndex.npy', allow_pickle=False)	
	#---Read length
	n = args.Read_length
	if(n != 100 and n != 150):
		print('ERROR: Invalid read lengths, please check help.')
		sys.exit()	
	#---maximum permissible error
	e = args.Error
	if(e < 0 or e > 8):
		print('ERROR: Permissible error out of range:[0-7], please check help.')
		sys.exit()

	#-- q-gram length------
	q_len = args.klen
	max_klen = math.floor(n/(e+1))
	#print(max_klen)
	if(q_len > max_klen or q_len < 12):
		print('ERROR: K-mer length not acceptable for pigeonhole principle. Or it should be within range:[12-25], Default: 12, please check help.')
		sys.exit()
	#---Maximum number of mappings allowed per read, determines the memory usage on the device. If not enough memory available, reduce this number.
	no_of_outputs_per_read = args.Outputs
	if(no_of_outputs_per_read > 3500 or no_of_outputs_per_read < 1):
		print('ERROR: Number of outputs desired per read out of range: [1:3500], please check help')
		sys.exit()
	#---Device number 
	if(args.dc):
		device_choices = args.dc; #print(device_choices)
	else:
		print('ERROR: Device choice/s not provided.')
		sys.exit()
	#---Print OpenCL version
	print('{:<30}'.format('PyOpenCL version'),':',cl.VERSION_TEXT)	
	#---Obtain reads from fastq file
	sequences, qualities, sequence_names = readFastq(p + args.Fastq_filename)
	#---Number of reads to map
	if(args.nr):
		no_of_reads = args.nr
	else:
		no_of_reads = len(sequences)
	sequences = sequences[0:no_of_reads]
	# read_sequence = sequences
	# sequences = ''.join(sequences)
	# sequences = np.array(sequences,'c')	

	#---Share per device
	if(args.rs):
		if(no_of_reads == sum(args.rs)):
			share_per_device = args.rs
		else:
			print('ERROR: Read shares do not sum upto the total number of reads')
			sys.exit()
	else:
		share_per_device = [no_of_reads]
		# print('ERROR: Read share among the devices required or the number of device selected do not have corresponding read share values available')
		# sys.exit()

	for i in share_per_device:
		if(str(i-int(i))[2:]):
			if(int(str(i-int(i))[2:]) != 0):
				print('ERROR: Read share is not leading integer distribution. Please enter correct proportion of read share or number of reads.')
				sys.exit()

	temp_dict = dict()
	temp_reads_alloc = 0
	temp_list = []
	reads = dict()
	for i in range(len(device_choices)):
		temp_dict[device_choices[i]] = int(share_per_device[i])
		temp_list = sequences[temp_reads_alloc:temp_reads_alloc + int(share_per_device[i])]
		temp_reads_alloc += int(share_per_device[i])
		temp_list = ''.join(temp_list)
		temp_list = np.array(temp_list,'c')
		reads[device_choices[i]] = temp_list
		
	share_per_device = temp_dict;
	del temp_dict; del temp_reads_alloc; del temp_list	

	total_memory_required = SA.nbytes + tally.nbytes + F.nbytes + sys.getsizeof(sequences) + no_of_reads*4 + no_of_reads*no_of_outputs_per_read*5
	print('{:<30}{:^3}{:.2f}'.format('Memory usage by datastructure',':',(SA.nbytes + tally.nbytes + F.nbytes)/(1024*1024*1024)),'GB')
	#-------------------------------------------------------------------
	strand_of_read, mapped_endpos_read, cand_locs_per_read = opencl(genome, reads, n, e, device_choices, no_of_reads, no_of_outputs_per_read, tally, SA, F, total_memory_required, share_per_device, q_len)
	print('-----------------------------------------------------------------------')
	print('Writing output to file')
	Write_Output(cand_locs_per_read, strand_of_read, mapped_endpos_read, device_choices, sequences, sequence_names, no_of_outputs_per_read, args.Fastq_filename.split('.')[0], n)
	print('-----------------------------------------------------------------------')

if __name__ == '__main__':
	main()

