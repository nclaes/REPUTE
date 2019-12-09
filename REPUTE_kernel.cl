
//  USING BANDED MYERS BIT VECTOR ALGORITHM
#define READ_LENGTH RLEN 
#define W WORD_LENGTH   //W is the WORD_LENGTH
#define ERROR PERMISSIBLE_ERROR
#define CAND_LOC_PER_READ CANDIDATES_PER_READ
#define MASK UINT_WITH_MSB_ONE // 32-bit unsigned integer with MSB as '1' rest all bits '0'
#define MIN_QG_LEN	MIN_QGRAM_LEN
#define EVAL_ARR_SIZE (READ_LENGTH - ((ERROR+1)*MIN_QG_LEN) + 1) + 1
#define NO_OF_ALPHABETS 4
#define NO_OF_ALPHABETS_INC_N 5
#define BAND_LEN READ_LENGTH + 2*ERROR


__kernel void repute(__global uchar* genome, __global char* Reads, __global uint* SA, __global uint* tally, __global uint* F,__global uint* cand_locs_per_read, __global uchar* genomic_strand_and_ED_for_mapped_reads, __global uint* endpos_for_mapped_reads)
{	

	char R[READ_LENGTH];  // R - Read
	uint local_F[NO_OF_ALPHABETS_INC_N];   // This should be 5 (no of alphabets + 1)
	int i=0,j=0,k=0,beginning=0,extent=0, current_prefix_pos=0, C = ERROR + 1, CONSTANT1 = W - 1 + READ_LENGTH - C;
	uint B_P[NO_OF_ALPHABETS_INC_N] = {0,0,0,0,0}, B[NO_OF_ALPHABETS_INC_N] = {0,0,0,0,0};  //B_P - B array Permanent used for both forward and reverse biased in Myers bit-vector algorithm
	uint X, D0, HP, HN, VP, VN, sa_start_pos, verif_start_pos_in_genome, occurences,no_of_locations=0;
	int edit_dist, temp_location, score,s=0;
	uint gid = get_global_id(0)*READ_LENGTH, last_successful_location=0;
	uint current_opt_feq[EVAL_ARR_SIZE], prev_opt_freq[EVAL_ARR_SIZE];
	uchar opt_div_pos[ERROR][EVAL_ARR_SIZE-1];
	uchar opt_div[ERROR + 2];

	for(j=0; j < READ_LENGTH; j++)
	{				
		switch(Reads[gid + j])
		{
			case 'T':
				R[j] = 3;
				break;
			case 'C':
				R[j] = 1;
				break;
			case 'G':
				R[j] = 2;
				break;
			default:  //case 'A': 'N' is also being treated as 'A'
				R[j] = 0;
				break;
		}
	}	
	for(j=0; j < C; j++) //preprocessing for banded Myers bit-vector algorithm
	{
		B_P[R[j]] = B_P[R[j]] | (1 << (W - C + j));
	}
	
	for(j=0; j < NO_OF_ALPHABETS + 1; j++)
	{
		local_F[j] = F[j];
	}
	for(k = 0; k < EVAL_ARR_SIZE; k++)
	{
		prev_opt_freq[k] = 0;
		current_opt_feq[k] = 0;
	}
	for(k = 0; k < ERROR; k++)
	{
		for(i = 0; i < EVAL_ARR_SIZE-1; i++)
		{
			opt_div_pos[k][i] = 0;
		}
	}
	// Filtration	
	beginning = READ_LENGTH - (ERROR*MIN_QG_LEN) - 1;	
	for(beginning = READ_LENGTH - (ERROR*MIN_QG_LEN) - 1; beginning > MIN_QG_LEN - 2; beginning--)
	{
		sa_start_pos = local_F[R[beginning]];
		occurences = local_F[R[beginning] + 1] - local_F[R[beginning]];
		for(current_prefix_pos = beginning-1; current_prefix_pos >= 0; current_prefix_pos--)
		{
			temp_location = tally[(sa_start_pos -1)*NO_OF_ALPHABETS + R[current_prefix_pos]];
			occurences = tally[(sa_start_pos + occurences - 1)*NO_OF_ALPHABETS + R[current_prefix_pos]] - temp_location;
			sa_start_pos = local_F[R[current_prefix_pos]] + temp_location;
			if(occurences == 0)
				{break;}
		}
		current_opt_feq[beginning - (MIN_QG_LEN-1)] = occurences;
		
	}
	
	for(i = ERROR - 1; i >= 0; i--)
	{
		for(k = 0; k < EVAL_ARR_SIZE; k++)
		{
			prev_opt_freq[k] = current_opt_feq[k];
			current_opt_feq[k] = 0;
		}
		extent = ((ERROR - i + 1) * MIN_QG_LEN) - 1;
		beginning = READ_LENGTH - (i * MIN_QG_LEN) - 1;
		edit_dist = beginning - MIN_QG_LEN + 1; // edit_dist is being used for current_opt_div (reusing variables)
		score = 0; //score is being used for optimal 2nd part frequency (reusing variables)

		for(j = beginning; j >= extent; j--)
		{
			s = score; // s is being used for previous optimal 2nd part frequency (reusing variable)
			verif_start_pos_in_genome = edit_dist; // verif_start_pos_in_genome is being used for previous optimal divider (reusing variables)
			sa_start_pos = local_F[R[j]];
			occurences = local_F[R[j] + 1] - local_F[R[j]];

			for(current_prefix_pos = j-1; current_prefix_pos > extent - MIN_QG_LEN; current_prefix_pos--)
			{
				X = current_prefix_pos - extent + MIN_QG_LEN; // X is being used for reference (reusing variables)
				D0 = occurences; // D0 is being used for prev_occurences (reusing variables)
				temp_location = tally[(sa_start_pos -1)*NO_OF_ALPHABETS + R[current_prefix_pos]];
				occurences = tally[(sa_start_pos + occurences - 1)*NO_OF_ALPHABETS + R[current_prefix_pos]] - temp_location;
				sa_start_pos = local_F[R[current_prefix_pos]] + temp_location;

				if(current_prefix_pos == j - MIN_QG_LEN + 1)
				{
					HP = prev_opt_freq[X - 1] + occurences; // HP is being used to store min_freq 
					edit_dist = current_prefix_pos;
					score = occurences;
				}
				else if(current_prefix_pos == verif_start_pos_in_genome && s == occurences)
				{
					edit_dist = verif_start_pos_in_genome;
					HP = prev_opt_freq[X - 1] + occurences;
					break;
				}
				else if(current_prefix_pos < j - MIN_QG_LEN + 1)
				{
					if(prev_opt_freq[X - 1] + occurences <= HP)
					{
						HP = prev_opt_freq[X - 1] + occurences;
						edit_dist = current_prefix_pos;
						score = occurences;
					}
					if(prev_opt_freq[X] - prev_opt_freq[X+1] > D0)
					{
						break;
					}
				}
			}
			opt_div_pos[i][j - extent] = edit_dist;
			current_opt_feq[j - extent] = HP;
			if(i == 0)
			{
				break;
			}
		}
	}


	opt_div[0] = READ_LENGTH;
	// printf("%u  ;;;;", opt_div[0]);
	opt_div[ERROR + 1] = 0;
	for(k = 0; k < ERROR; k++)
	{
		extent = ((ERROR - k + 1)* MIN_QG_LEN) - 1;
		opt_div[k+1] = opt_div_pos[k][opt_div[k] - extent - 1];
	}



	gid = get_global_id(0)*CAND_LOC_PER_READ;
	i=0; j=0; k=0; beginning=0; extent=0; current_prefix_pos=0;
	X=0; D0=0;  sa_start_pos=0; occurences=0; s=0; score = 0;
	for(i = 0; i <= ERROR; i++)
	{
		sa_start_pos = local_F[R[opt_div[i]-1]];
		occurences = local_F[R[opt_div[i]-1] + 1] - local_F[R[opt_div[i]-1]];
		for(current_prefix_pos = opt_div[i]-2; current_prefix_pos >= opt_div[i+1]; current_prefix_pos--)
		{
			temp_location = tally[(sa_start_pos -1)*NO_OF_ALPHABETS + R[current_prefix_pos]];
			occurences = tally[(sa_start_pos + occurences - 1)*NO_OF_ALPHABETS + R[current_prefix_pos]] - temp_location;
			sa_start_pos = local_F[R[current_prefix_pos]] + temp_location;
		}	
		// occurences = (occurences > 1000)?1000:occurences;
		for(j = 0; j < occurences; j++)
		{	
			verif_start_pos_in_genome = SA[sa_start_pos + j] - ERROR - opt_div[i+1];
			if(no_of_locations >= CAND_LOC_PER_READ || (verif_start_pos_in_genome > last_successful_location && verif_start_pos_in_genome < last_successful_location  + 4*ERROR))
			{
				// number of locations is GREATER than CANDIDATES_PER_READ	
				continue;
			}					
			score = C;   // Reseting of score
			edit_dist = ERROR+1;
			VP = ~0; VN = 0;					
			B[0] = B_P[0]; B[1] = B_P[1]; B[2] = B_P[2];B[3] = B_P[3]; B[4] = B_P[4]; 	
			for(k = 0; k < BAND_LEN; k++) // verifying for n+2e length    	READ_LENGTH + ERROR + ERROR
			{				
				B[0] = B[0] >> 1;
				B[1] = B[1] >> 1;
				B[2] = B[2] >> 1;
				B[3] = B[3] >> 1;
				B[4] = B[4] >> 1;
				if(k + C < READ_LENGTH)
				{
					B[R[k+C]] = B[R[k+C]] | MASK;
				}
				X = B[genome[k + verif_start_pos_in_genome]] | VN;
				D0 = ((VP + (X & VP)) ^ VP) | X;
				HN = VP & D0;
				HP = VN | ~(VP | D0);
				X = D0 >> 1;
				VN = X & HP;
				VP = HN | ~(X | HP);

				if(k < (READ_LENGTH-C))
				{
					score = score + 1 - ((D0 >> (W-1)) & 1);
				}
				else
				{
					s = CONSTANT1 - k;//s = (W-2) - (k - (READ_LENGTH - C + 1));	
					score = score + ((HP >> s) & 1);
					score = score - ((HN >> s) & 1);
				}
				if(score < edit_dist && (k >= (READ_LENGTH-C)))
				{
					edit_dist = score;
					temp_location = k + verif_start_pos_in_genome;
				}
			}
			if(edit_dist <= ERROR)
			{
				last_successful_location = temp_location - (READ_LENGTH + 2*ERROR);
				endpos_for_mapped_reads[gid + no_of_locations] = temp_location+1;			//Adding 1 as the location is zero-based
				genomic_strand_and_ED_for_mapped_reads[gid + no_of_locations] = 128 + edit_dist;
				no_of_locations = no_of_locations + 1;					
			}				
		}		
	}



	// // REVERSE STRAND
	gid = get_global_id(0)*READ_LENGTH;
	i=0; j=0; k=0; beginning=0; extent=0; current_prefix_pos=0;
	X=0; D0=0; HP=0; HN=0; sa_start_pos=0; occurences=0; 
	edit_dist = 0; temp_location = 0; verif_start_pos_in_genome = 0; score = 0; s=0;
	
	for(i=0; i < READ_LENGTH; i++)
	{				
		switch(Reads[gid + i])
		{
			case 'T':
				R[READ_LENGTH-1-i] = 0;
				break;
			case 'C':
				R[READ_LENGTH-1-i] = 2;
				break;
			case 'G':
				R[READ_LENGTH-1-i] = 1;
				break;
			default://case 'A' and 'N'   in reverse strand N is being treated as T
				R[READ_LENGTH-1-i] = 3;
				break;
		}		
	}
	for(j=0; j<NO_OF_ALPHABETS_INC_N; j++)
	{
		B_P[j] = 0;
	}
	for(j=0; j < C; j++) //preprocessing for banded Myers bit-vector algorithm
	{		
		B_P[R[j]] = B_P[R[j]] | (1 << (W - C + j));
	}
	for(k = 0; k < ERROR+2; k++)
	{
		opt_div[k] = 0;
	}

	for(k = 0; k < EVAL_ARR_SIZE; k++)
	{
		prev_opt_freq[k] = 0;
		current_opt_feq[k] = 0;
	}

	for(k = 0; k < ERROR; k++)
	{
		for(i = 0; i < EVAL_ARR_SIZE-1; i++)
		{
			opt_div_pos[k][i] = 0;
		}
	}

	beginning = READ_LENGTH - (ERROR*MIN_QG_LEN) - 1;
	for(beginning = READ_LENGTH - (ERROR*MIN_QG_LEN) - 1; beginning > MIN_QG_LEN - 2; beginning--)
	{
		sa_start_pos = local_F[R[beginning]];
		occurences = local_F[R[beginning] + 1] - local_F[R[beginning]];
		for(current_prefix_pos = beginning-1; current_prefix_pos >= 0; current_prefix_pos--)
		{
			temp_location = tally[(sa_start_pos -1)*NO_OF_ALPHABETS + R[current_prefix_pos]];
			occurences = tally[(sa_start_pos + occurences - 1)*NO_OF_ALPHABETS + R[current_prefix_pos]] - temp_location;
			sa_start_pos = local_F[R[current_prefix_pos]] + temp_location;
			if(occurences == 0)
				{break;}
		}
		current_opt_feq[beginning - (MIN_QG_LEN-1)] = occurences;
	}	
	
	for(i = ERROR - 1; i >= 0; i--)
	{
		for(k = 0; k < EVAL_ARR_SIZE; k++)
		{
			prev_opt_freq[k] = current_opt_feq[k];
			current_opt_feq[k] = 0;
		}
		extent = ((ERROR - i + 1) * MIN_QG_LEN) - 1;
		beginning = READ_LENGTH - (i * MIN_QG_LEN) - 1;
		edit_dist = beginning - MIN_QG_LEN + 1; // edit_dist is being used for current_opt_div (reusing variables)
		score = 0; //score is being used for optimal 2nd part frequency (reusing variables)
		for(j = beginning; j >= extent; j--)
		{
			s = score; // s is being used for previous optimal 2nd part frequency (reusing variable)
			verif_start_pos_in_genome = edit_dist; // verif_start_pos_in_genome is being used for previous optimal divider (reusing variables)
			sa_start_pos = local_F[R[j]];
			occurences = local_F[R[j] + 1] - local_F[R[j]];

			for(current_prefix_pos = j-1; current_prefix_pos > extent - MIN_QG_LEN; current_prefix_pos--)
			{
				X = current_prefix_pos - extent + MIN_QG_LEN; // X is being used for reference (reusing variables)
				D0 = occurences; // D0 is being used for prev_occurences (reusing variables)
				temp_location = tally[(sa_start_pos -1)*NO_OF_ALPHABETS + R[current_prefix_pos]];
				occurences = tally[(sa_start_pos + occurences - 1)*NO_OF_ALPHABETS + R[current_prefix_pos]] - temp_location;
				sa_start_pos = local_F[R[current_prefix_pos]] + temp_location;

				if(current_prefix_pos == j - MIN_QG_LEN + 1)
				{
					HP = prev_opt_freq[X - 1] + occurences; // HP is being used to store min_freq 
					edit_dist = current_prefix_pos;
					score = occurences;
				}
				else if(current_prefix_pos == verif_start_pos_in_genome && s == occurences)
				{
					edit_dist = verif_start_pos_in_genome;
					HP = prev_opt_freq[X - 1] + occurences;
					break;
				}
				else if(current_prefix_pos < j - MIN_QG_LEN + 1)
				{
					if(prev_opt_freq[X - 1] + occurences <= HP)
					{
						HP = prev_opt_freq[X - 1] + occurences;
						edit_dist = current_prefix_pos;
						score = occurences;
					}
					if(prev_opt_freq[X] - prev_opt_freq[X+1] > D0)
					{
						break;
					}
				}
			}
			opt_div_pos[i][j - extent] = edit_dist;
			current_opt_feq[j - extent] = HP;
			if(i == 0)
			{
				break;
			}
		}
	}

	opt_div[0] = READ_LENGTH;
	opt_div[ERROR + 1] = 0;
	for(k = 0; k < ERROR; k++)
	{
		extent = ((ERROR - k + 1)* MIN_QG_LEN) - 1;
		opt_div[k+1] = opt_div_pos[k][opt_div[k] - extent - 1];
	}

	gid = get_global_id(0)*CAND_LOC_PER_READ;
	i=0; j=0; k=0; beginning=0; extent=0; current_prefix_pos=0;
	X=0; D0=0;  sa_start_pos=0; occurences=0; s=0; score = 0;
	for(i = 0; i <= ERROR; i++)
	{
		sa_start_pos = local_F[R[opt_div[i]-1]];
		occurences = local_F[R[opt_div[i]-1] + 1] - local_F[R[opt_div[i]-1]];
		
		for(current_prefix_pos = opt_div[i]-2; current_prefix_pos >= opt_div[i+1]; current_prefix_pos--)
		{
			temp_location = tally[(sa_start_pos -1)*NO_OF_ALPHABETS + R[current_prefix_pos]];
			occurences = tally[(sa_start_pos + occurences - 1)*NO_OF_ALPHABETS + R[current_prefix_pos]] - temp_location;
			sa_start_pos = local_F[R[current_prefix_pos]] + temp_location;
		}
		// occurences = (occurences > 1000)?1000:occurences;
		for(j = 0; j < occurences; j++)
		{	
			verif_start_pos_in_genome = SA[sa_start_pos + j] - ERROR - opt_div[i+1];
			if(no_of_locations >= CAND_LOC_PER_READ || (verif_start_pos_in_genome > last_successful_location && verif_start_pos_in_genome < last_successful_location  + 4*ERROR))
			{	
				continue;
			}					
			score = C;   // Reseting of score
			edit_dist = ERROR+1;
			VP = ~0; VN = 0;
			B[0] = B_P[0]; B[1] = B_P[1]; B[2] = B_P[2];B[3] = B_P[3]; B[4] = B_P[4];
			for(k = 0; k < BAND_LEN; k++) // verifying for n+2e length 
			{
				B[0] = B[0] >> 1;
				B[1] = B[1] >> 1;
				B[2] = B[2] >> 1;
				B[3] = B[3] >> 1;
				B[4] = B[4] >> 1;
				if(k + C < READ_LENGTH)
				{
					B[R[k+C]] = B[R[k+C]] | MASK;
				}
				X = B[genome[k + verif_start_pos_in_genome]] | VN;
				D0 = ((VP + (X & VP)) ^ VP) | X;
				HN = VP & D0;
				HP = VN | ~(VP | D0);
				X = D0 >> 1;
				VN = X & HP;
				VP = HN | ~(X | HP);

				if(k < (READ_LENGTH-C))
				{
					score = score + 1 - ((D0 >> (W-1)) & 1);
				}
				else
				{
					s = CONSTANT1 - k;//s = (W-2) - (k - (READ_LENGTH - C + 1));	
					score = score + ((HP >> s) & 1);
					score = score - ((HN >> s) & 1);
				}
				if(score < edit_dist && (k >= (READ_LENGTH-C)))
				{
					edit_dist = score;
					temp_location = k + verif_start_pos_in_genome;
				}
			}					
			if(edit_dist <= ERROR)
			{
				last_successful_location = temp_location - (READ_LENGTH + 2*ERROR);
				endpos_for_mapped_reads[gid + no_of_locations] = temp_location+1;
				genomic_strand_and_ED_for_mapped_reads[gid + no_of_locations] = edit_dist;
				no_of_locations = no_of_locations + 1;
			}				
		}			
	}
	cand_locs_per_read[get_global_id(0)] = no_of_locations;
	barrier(CLK_GLOBAL_MEM_FENCE);
}
