#!/bin/bash

##Kø navn, fx hpc som er den generelle cpu. Der er også gpu køer
#BSUB -q gpuv100
##Antal gpuer vi vil bruge. Kommenter ud hvis cpu.
#BSUB -gpu "num=1:mode=exclusive_process"

##Navn på job. Hvis man vil lave mange jobs kan man skrive my_job_name[1-100] så får man 100 jobs.
#BSUB -J PI_WGAN_test
##Output log fil. Folderen skal eksistere før jobbet submittes. Job nummer indsættes automatisk ved %J i filnavnet.
#BSUB -o output/PI_WGAN_test-%J.out
##Antal cpu kerner
#BSUB -n 1
##Om kernerne må være på forskellige computere
#BSUB -R "span[hosts=1]"
##Ram pr kerne
#BSUB -R "rusage[mem=8GB]"
##Hvor lang tid må den køre hh:mm
#BSUB -W 23:59
##Modtag email på studiemail når jobbet starter
#BSUB -B
##mail når jobbet stopper
#BSUB -N

## Fjern alle pakker og load så dem man skal bruge. Brug "module avail" på dtu server for at se hvilke pakker der findes.
module purge

module load python3/3.10.2
module load cuda/11.3
module load cudnn/v8.2.0.53-prod-cuda-11.3
module load matplotlib/3.5.1-numpy-1.22.2-python-3.10.2


## Konstant argument til programmet

## Hvis man skal lave et loop hvor programmet modtager forskellige argumenter.
for n_epochs in 1000 10000
do
	for weigth_clipping in 0 1
	do
		for GP_on in 0 1
		do 
			for n_sols in 100 20000
			do
				for optimizer_choice in 0 1
				do
					python3 PIG_WGAN.py --n_epochs=$n_epochs --weigth_clipping=$weigth_clipping --GP_on=$GP_on --n_sols=$n_sols --optimizer_choice=$optimizer_choice
				done
			done
		done
	done
done

## Har man oprettet jobbet som en liste af jobs my_job_name[1-100] så kan man bruge dette indeks fra 1 til 100 som argument til sit program med argumentet $LSB_JOBINDEX
