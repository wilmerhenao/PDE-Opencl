#! /bin/bash
set -e

function iloop()
{
  for points in 33 65 129; do
    make clean > /dev/null
    make mg DEFINES="$ufourth $ualign $ud -DDO_TIMING -DPOINTS=$points -DVERSION=$version -DCUTOFF=$cutoff" > /dev/null
    ./mg
  done
  echo
}

function iloopversion2()
{
  for points in 33 65 129; do
    make clean > /dev/null
    make mg2 DEFINES="$ufourth $ualign $ud -DDO_TIMING -DPOINTS=$points -DVERSION=$version -DCUTOFF=$cutoff" > /dev/null
    ./mg2
  done
  echo
}


  ufourth=""
  for ud in "" "-DUSE_DOUBLE" ; do
    for ualign in "-DUSE_ALIGNMENT" ; do
      for version in 0 1 2 3 4 5 ; do
	  for cutoff in 16 ; do
	     iloopversion2
	  done
      done
    done
  done

  echo "running scenarios"

  ufourth=""
  for ud in "" "-DUSE_DOUBLE" ; do
    for ualign in "-DUSE_ALIGNMENT" ; do
      for version in 0 1 2 3 4 5; do
	  for cutoff in 16 ; do
	     iloop
	  done
      done
    done
  done
