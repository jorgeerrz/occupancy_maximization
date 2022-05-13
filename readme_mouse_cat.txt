To generate the data, scripts with following filenames were used

predator-prey scenario: gridworld_R for R agent, gridworld_H for H agent
friendly pet scenario: gridfriendly_poly

To run a script, one need python installed, and run the command line: 

python filename.txt 

with the respective filename.


Parameters of the simulations are just variables in the script files, particularly

F   is   energy limit,
gamma - discount coeffitient for the optimized function,
N - arena size (so that arena is inside N by N square),
beta - inverted predator temperature  (kappa in the main text),
nz - the level of noise in action selection (epsilon for the epsilon-greedy policy in the main text),
steps - simulation length.

Arena shape is coded in the 2 by N by N array form, 1 in form[0,i,j] is for existance of a field at (i,j) location for the agent, and in form[1,i,j] for the predator. (In the friendly pet scenario form array is effectively splitted into form0 not taken fences into account and bordform, where bordform[a], N by an array, describes the shape of the fence number a.)
Food amount at (i,j) is given by food[i,j] in the N by N array food.   

If the variable uload is 1, then policy is read out from the file, and if uload is 0, the policy is first calculated, and also then saved in the file.
The name of the file with the policy is formed as "zformedcommonW_mouselab_bottleneck_fulltank " +"N"+str(N)+"F"+str(F)+"beta"+str(beta)+"miop"+str(miop)+"nz"+str(nz)+"has"+str(uhas)
Default values of the respective variables are N=7, F=15, beta=2., miop=0, nz=0. (or 0.05 for epsilon-greedy R agent), has=0

Other generated files have names of the same structure:
The generated video is saved in 'output_video_plots_'+pref+'steps'+str(steps)+'.mp4'

Occupancy averaged over simulation time - in "occ_"+pref+"steps"+str(steps)
Averaged frequencies of transitions from a given location to available directions- in "vocc_norm_"+pref+"steps"+str(steps)
Numbers of full rotations around the wall - in "rot2co_"+pref+"_steps"+str(steps)
Averaged lifetimes of the agent - in "tlifes_"+pref+"_steps"+str(steps)
Times and number of rotations ordered in time - in "trot_"+pref+"_steps"+str(steps)
Trajectory - in "rec_"+pref+"_steps"+str(steps)
And for friendly pet scenario also the summed time with open/closed fence - in "cc0_"+str(nk)+"x"+str(ncop)+"_"+pref with ncop=10 staying for number of simulations with same parameters and nk=11 for number of parameter values

	


