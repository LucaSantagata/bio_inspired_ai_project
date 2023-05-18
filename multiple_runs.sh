#!/bin/bash

while getopts R:r:dpP:vV:st flag
do
    case "${flag}" in
        R) runs=${OPTARG};;
        r) runatatime=${OPTARG};;
        d) default="default";;
        P) savepop=${OPTARG};;
        p) dp="savepop";;
        V) savevideo=${OPTARG};;
        v) dv="savevideo";;
        s) seed="seed";;
        t) test=${OPTARG};;
    esac
done

#echo "$runs , $default , $savepop , $dp , $savevideo , $dv , $seed , $test"

rand=$RANDOM
default_savepop="./save_pop"
default_savevideo="./save_video"

for i in {1..$[$runs/$runatatime]}
do
#  echo "$i"

  cmd="python ./PyGenoCar.py --run "
  cmd+="$i"

  params="_"
  params+="$rand"

#  if [[ ! -z $default ]]
#    then
##      echo "default"
#  fi

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

  if [[ ! -z "$test" ]]
    then
#      echo "test $test"
      params+=" --test-from-filename $test"
  fi

  if [[ ! -z "$seed" ]]
    then
#      echo "seed $rand"
      params+=" --seed $rand"
  fi

  echo "$cmd-x$params"

  finalcmd=""
  for j in {1..$runatatime}
    do
      finalcmd+="($cmd-$j$params) & "
  done

  echo "$finalcmd wait"
#  eval "($cmd-a$params) & ($cmd-b$params) & ($cmd-c$params) & wait"
  eval "$finalcmd wait"
done
