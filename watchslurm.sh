#!/bin/bash

## This utility monitors my slurm jobs as well as my pending jobs
## by using an auxiliary script getslurmjobs
watch -t "./getslurmjobs running; ./getslurmjobs pending"
