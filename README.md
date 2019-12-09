# REPUTE
REad maPper  for  heterogeneoUs sysTEms

REPUTE is an OpenCL based read aligner. It is like an all-mapper and reports upto a specified number of mappings per read. It pre-processes the genome/genomic_section/chromosome using FM-Index and suffix array to produce the datastructure files to be used while mapping reads. It employs pigeonhole principle combined with dynamic programming based k-mer/seed selection criteria inspired from the Optimum Seed Solver[1]. Dynamic programming based filteration explores all possible lengths of k-mers, within given constraints, to reduce the total number of candidate locations. Using this filteration scheme it reduces the total number of verification steps significanlty, therefore, producing enhanced performance. 

REPUTE can be used on any OpenCL conformant device. The host program is written in Python while the OpenCL kernel for parallel computation is written in C. We have used REPUTE on four different systems with different combinations, versions and technology of CPU, including Intel Core i5, i7 and GPU, including Nvidia GTX490 and Tesla C1060. For detailed results please refer to the manuscript accepted in Design, Automation and Test in Europe Conference 2020 (in press) [2].  

REPUTE has been divided into four parts: REPUTE_preprocessing.py, REPUTE_get_device_choice.py, REPUTE_host.py and REPUTE_kernel.cl

I) REPUTE_preprocessing.py -- Usage instructions can be requested using the following:
   
    python3 REPUTE_preprocessing.py -h 
    python3 REPUTE_preprocessing.py chr2.fa
   
The input accepts a fasta file and produces three datastructure files viz. Tally.npy, SA.npy, F_FMIndex.npy. The overall memory required for the detastructure depends on the size of the genome. We use online available programs with reusablity license to obtain suffix array of any genome. Source code and License files can be found in the project folder.

II) REPUTE_get_device_choice.py -- Usage:

    python3 REPUTE_get_device_choice.py 
    
    Example output:
    Following is the list of platforms:
      PLATFORM NO.    : PLATFORM NAME/S

       0        : [<pyopencl.Device 'Intel(R) Core(TM) i7-2600 CPU @ 3.40GHz' on 'Intel(R) OpenCL' at 0x270b348>] 

       1        : [<pyopencl.Device 'GeForce GTX 590' on 'NVIDIA CUDA' at 0x27212a0>, <pyopencl.Device 'Tesla C1060' on 'NVIDIA CUDA' at 0x275a060>, <pyopencl.Device 'GeForce GTX 590' on 'NVIDIA CUDA' at 0x2254df0>] 

      ----------------------------------------------------------------------

      Following is the list of all devices:
      DEVICE NO.      : DEVICE NAME

             0        : <pyopencl.Device 'Intel(R) Core(TM) i7-2600 CPU @ 3.40GHz' on 'Intel(R) OpenCL' at 0x270b348> 

             1        : <pyopencl.Device 'GeForce GTX 590' on 'NVIDIA CUDA' at 0x27212a0> 

             2        : <pyopencl.Device 'Tesla C1060' on 'NVIDIA CUDA' at 0x275a060> 

             3        : <pyopencl.Device 'GeForce GTX 590' on 'NVIDIA CUDA' at 0x2254df0> 

      ----------------------------------------------------------------------
    
As can be seen from an example output, the system under consideration has 2 platforms viz. Intel and Nvidia. And four devices altogether viz. two GTX590, one C1060 and one Intel quad-core i7 CPU. The device choices are required to be used when running the host program. This indicates on which devices does the user want to map reads. Note that installation of OpenCL SDK or drivers are required to obtain device choices. It, also, confirms the correct installation of the drivers. 

III) REPUTE_host.py- Usage:

	python3 REPUTE_host.py -h
	-----------------------------------------------------------------------
	usage: REPUTE_host.py [-h] [-dc DC [DC ...]] [-rs RS [RS ...]] [-nr NR]
              Fastq_filename Read_length Error Outputs klen

	REPUTE : REad maPper for heterogeneoUs sysTEms . This tool maps short-reads to
	reference genome.

	positional arguments:
	  Fastq_filename   Give the name of reads file in fastq format.
	  Read_length      Provide the length of the reads. Choices: 100 or 150
	  Error            Provide the maximum permissible error. Smaller the value
	                   faster will be the algorithm. Range:[0-8]
	  Outputs          Maximum number of mappings allowed per read. Range:[1:3500]
	  klen             K-mer length. Range:[12:25]

	optional arguments:
	  -h, --help       show this help message and exit
	  -dc DC [DC ...]  Provide the numbers of all the devices to be used. These
	                   numbers can be from running 'get_device_choice.py' script
	  -rs RS [RS ...]  Provide the share of reads to be mapped for each device,
	                   with the sum accumulating to the total number of reads e.g.
	                   for a given 100000 should lead to a share of [50000 30000
	                   20000] for three devices. All shares MUST be integers.
	                   Number of reads in the multiples of 2 will lead to better
	                   performance, especially for GPUs.
	  -nr NR           Number of reads to be mapped, default: all the reads in the
	                   fastq file

The device choices obtained with the 'REPUTE_get_device_choice.py' script is used to specify the devices to be using for read mapping. Along with the devices, user is needed to specify the workload, i.e. number of reads, to be processed by each device chosen. 

      Example:
      python3 REPUTE_host.py SRR826460_1_1000000_reads.fq 150 7 1000 -dc 0 1 3 -rs 488000 256000 256000
      
      Here, read length is 150, permissible number of errors, i.e. edit distance, 7, chosen devices 
      are the Intel quad-core CPU and two Nvidia GTX 590s. The read distribution is 488000 to CPU and 256000 each to
      the two Nvidia GPUs.

IV) REPUTE_kernel.cl : This is the OpenCL kernel file which performs filtration and verifications steps of read mapping for all reads.
      
Prerequisites: Installation of OpenCL SDK and drivers for all the platforms available on the system. Python and PyOpenCL are also required to be installed on the systems. 

Note: Any problem while running REPUTE will mainly be due to incorrect installations or memory allocation based issue. There should be sufficient memory to allocate for the data structure. OpenCL permits a maximum of 1/4th of the total RAM available on the device to any single variable. Thus, in case of large chromosomes such as chr2, it is advised to have atleast 16GB of RAM installed on the device. This limitation will be removed from future versions of the tool. If suftest executable throws error, we will recommend extracting the SA.zip and recompiling it. Then overwrite the old executable with the new one. 


[1] H. Xin, S. Nahar, R. Zhu, J. Emmons, G. Pekhimenko, C. Kingsford, C. Alkan, and O. Mutlu, “Optimal seed solver: optimizing seed selection in read mapping,” Bioinformatics, vol. 32, no. 11, pp. 1632–1642, 2016.

[2] S. Maheshwari, R. Shafik, I. Wilson, A. Yakovlev, and A. Acharyya, “REPUTE: An OpenCL based Read Mapping Tool for Embedded Genomics," Design, Automation and Test in Europe (DATE) 2020 (in press).

[3] S. Maheshwari, V. Y. Gudur, R. Shafik, I. Wilson, A. Yakovlev, and A. Acharyya, “Coral: Verification-aware opencl based read mapper for heterogeneous systems,” IEEE/ACM Transactions on Computational
Biology and Bioinformatics, pp. 1–1, 2019.