#!/bin/bash

while getopts R:dpP:vV:st flag
do
    case "${flag}" in
        R) run=${OPTARG};;
        d) default="default";;
        P) savepop=${OPTARG};;
        p) dp="savepop";;
        V) savevideo=${OPTARG};;
        v) dv="savevideo";;
        s) seed="seed";;
        t) test=${OPTARG};;
    esac
done

#echo "$run , $default , $savepop , $dp , $savevideo , $dv , $seed , $test"

rand=$RANDOM
default_savepop="./save_pop"
default_savevideo="./save_video"

for i in {1..$run}
do
  echo "$i"

  cmd="python ./PyGenoCar.py --run "
  cmd+="$i"
  cmd+="_"
  cmd+="$rand"

  if [[ ! -z $default ]]
    then
      echo "default"
  fi

  if [[ ! -z "$savepop" ]]
    then
      echo "savepop $savepop"
      cmd+=" --save-pop $savepop"
  fi

  if [[ ( ! -z "$default" ) && ( ! -z "$dp" ) ]]
    then
      echo "default"
      echo "savepop $default_savepop"
      cmd+=" --save-pop $default_savepop"
  fi

  if [[ ! -z "$savevideo" ]]
    then
      echo "savevideo $savevideo"
      cmd+=" --save-video $savevideo"
  fi

  if [[ ( ! -z "$default" ) && ( ! -z "$dv" ) ]]
    then
      echo "default"
      echo "savevideo $default_savevideo"
      cmd+=" --save-video $default_savevideo"
  fi

  if [[ ! -z "$test" ]]
    then
      echo "test $test"
      cmd+=" --test-from-filename $test"
  fi

  if [[ ! -z "$seed" ]]
    then
      echo "seed $rand"
      cmd+=" --seed $rand"
  fi

  echo "$cmd"
  eval "$cmd"
done
