import shutil
import sys
import fileinput
import numpy as np

B_list = [200*50, 200]
fips_list = [53033, 53047]
p_list = [0,1,2,3,4]
sa_list = [i for i in range(11)]
count=20

# for fips in fips_list:
#     for p in p_list:
#         org_sh = 'run.sh'
#         my_sh = 'alloc_'+str(count)+'.sh'
#         shutil.copy(org_sh, my_sh)
#         for line in fileinput.input(my_sh, inplace=1):
#             if "job-name" in line:
#                 curr = line
#                 line =  "#SBATCH --job-name=al_"+str(count)+"\n"
#             if "python " in line:
#                 curr = line
#                 line =  f"python alloc.py -f {fips} -p {p} -B {B_list[fips_list.index(fips)]}\n"
#             sys.stdout.write(line)
#         count+=1
for fips in fips_list:
    for sa in sa_list:
        org_sh = 'run.sh'
        my_sh = 'sa_'+str(count)+'.sh'
        shutil.copy(org_sh, my_sh)
        for line in fileinput.input(my_sh, inplace=1):
            if "job-name" in line:
                curr = line
                line =  "#SBATCH --job-name=sa_"+str(count)+"\n"
            if "python " in line:
                curr = line
                line =  f"python sa_alloc.py -f {fips} -p 0 -B {B_list[fips_list.index(fips)]} -i {sa} \n"
            sys.stdout.write(line)
        count+=1