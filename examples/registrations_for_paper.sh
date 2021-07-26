#!/bin/bash
time="50:00:00"
partition="parallel"
    
convert_to_fidis_dir () {
    fidis_dir=${1//mnt\/data\/FA/scratch\/aymanns}
    fidis_dir=${1//mnt\/data2\/FA/scratch\/aymanns}
    fidis_dir=${fidis_dir//mnt\/lab_server\/AYMANNS_Florian\/Experimental_data/scratch\/aymanns}
    echo ${fidis_dir}
}

while read dir; do
    #echo ${dir}
    folder=$(convert_to_fidis_dir ${dir})
    #if [[ "${folder}" =~ "023_coronal" ]] && [[ -e ${folder} ]] && [[ ! -f "${folder}/warped_red.tif" ]] && [[ ! -f "${folder}/w.npy" ]]; then
    if [[ -e ${folder} ]] && [[ ! -f "${folder}/warped_red.tif" ]] && [[ ! -f "${folder}/w.npy" ]]; then
        ref_frame="${folder}/ref_frame.tif"
        echo ${folder}
        echo ${ref_frame}
        sbatch --nodes 1 --ntasks 1 --cpus-per-task 28 --time "${time}" --mem 128G --partition ${partition} "registration_commands.sh" ${folder} ${ref_frame}
    fi
done <trials_for_paper.txt
