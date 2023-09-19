import shutil
import sys
import fileinput

B_list = [5000,1200,100]
fips_list = [53033,53011,53047]
p_list = [i for i in range(10)]
count=0

for k in range(len(B_list)):
    for p in p_list:
        org_sh = 'run.sh'
        my_sh = 'alloc_'+str(count)+'.sh'
        shutil.copy(org_sh, my_sh)
        for line in fileinput.input(my_sh, inplace=1):
            if "job-name" in line:
                curr = line
                line =  "#SBATCH --job-name=alloc_"+str(count)+"\n"
            if "python " in line:
                curr = line
                line =  "python alloc.py -B "+str(B_list[k])+" -p "+str(p)+" -f "+str(fips_list[k])+"\n"
            sys.stdout.write(line)
        count+=1