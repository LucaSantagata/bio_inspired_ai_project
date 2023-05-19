#!/bin/bash

help="Commands:
\t- R = number of total runs
\t- r = runs at a time
\t- d = default params and savepop
\t- p = change savepop directory
\t- v = default savevideo
\t- V = change savevideo directory
\t- s = generate random but static seed for all runs"

while getopts R:r:dp:vV:sH flag
do
    case "${flag}" in
        R) runs=${OPTARG};;
        r) runatatime=${OPTARG};;
        d) default="default";;
        p) savepop=${OPTARG};;
#        p) dp="savepop";;
        V) savevideo=${OPTARG};;
        v) dv="savevideo";;
        s) seed="seed";;
        H) echo "$help"; exit;;
        *) echo "Wrong param"; exit;;
    esac
done

#echo "$runs , $runatatime , $default , $savepop , $dp , $savevideo , $dv , $seed , $test"


rand=$RANDOM
default_savepop="./save_pop"
default_savevideo="./save_video"

i=1
while [ $i -le $((runs/runatatime)) ];
do
#  echo "$i"

  cmd="python ./PyGenoCar.py --run "
  cmd+="$i"

  params="_"
  params+="$rand"

  if [[ ! -z $default ]]
    then
#      echo "default"
      dp="savepop"
  fi

  if [[ ! -z "$savepop" ]]
    then
#      echo "savepop $savepop"
      params+=" --save-pop $savepop"
  fi

  if [[ ( ! -z "$default" ) && ( ! -z "$dp" ) ]]
    then
#      echo "default"
#      echo "savepop $default_savepop"
      params+=" --save-pop $default_savepop"
  fi

  if [[ ! -z "$savevideo" ]]
    then
#      echo "savevideo $savevideo"
      params+=" --save-video $savevideo"
  fi

  if [[ ( ! -z "$default" ) && ( ! -z "$dv" ) ]]
    then
#      echo "default"
#      echo "savevideo $default_savevideo"
      params+=" --save-video $default_savevideo"
  fi

  if [[ ! -z "$seed" ]]
    then
#      echo "seed $rand"
      params+=" --seed $rand"
  fi

  finalcmd=""
  j=1
  while [ $j -le $runatatime ];
    do
      finalcmd+="($cmd-$j$params) & "
      j=$((j+1))
  done

  echo "$finalcmd wait"
#  eval "($cmd-a$params) & ($cmd-b$params) & ($cmd-c$params) & wait"
  eval "$finalcmd wait"
  i=$((i+1))
done
