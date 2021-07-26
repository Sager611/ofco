#!/bin/sh

folder=$1
ref_frame=$2
module load gcc/7.4.0 python/3.7.3
source /home/aymanns/ofco/examples/venvs/venv-for-registration/bin/activate
echo STARTING AT `date`
python /home/aymanns/ofco/examples/register.py ${folder} ${ref_frame}
echo FINISHED at `date`
