SHIFTER=docker:ghcr.io/erykoff/redmapper:0.8.5.5

# salloc -N 1 -C cpu -A des -L cfs,SCRATCH -t 02:00:00 --qos interactive --image=docker:ghcr.io/erykoff/redmapper:0.8.5.5

# Calibrate
shifter --module=mpich --image docker:ghcr.io/erykoff/redmapper:0.8.5.5 /bin/bash -c ". /opt/redmapper/startup.sh && /global/homes/m/mkwiecie/des/repos/redmapper/bin/redmapper_calibrate.py -c cal.yml"

# ZRed
# Edit $HOME/.redmapper_batch.yaml then run below
shifter --module=mpich --image docker:ghcr.io/erykoff/redmapper:0.8.5.5 /bin/bash -c ". /opt/redmapper/startup.sh && /global/homes/m/mkwiecie/des/repos/redmapper/bin/redmapper_batch.py -c run.yml -b NERSC -N 1 -r 1"

#!/bin/sh
#SBATCH --job-name=y6_gold_2.0-3_wide_bdf_mktest_run_zred[0-136]
#SBATCH --nodes=1
#SBATCH --mem=2000
#SBATCH --array=0-136
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=5:00:00
#SBATCH --account=des
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --licenses=cfs,SCRATCH
#SBATCH --image=docker:ghcr.io/erykoff/redmapper:0.8.5.5
#SBATCH --output=/pscratch/sd/m/mkwiecie/redmapper_try1/run/jobs/y6_gold_2.0-3_wide_bdf_mktest_run_zred-%A-%a.log
#SBATCH --error=/pscratch/sd/m/mkwiecie/redmapper_try1/run/jobs/y6_gold_2.0-3_wide_bdf_mktest_run_zred-%A-%a.err

pixarr=(304 305 306 307 336 337 338 339 340 364 365 366 367 368 369 370 371 372 396 397 398 399 400 401 402 403 404 428 429 430 431 432 433 434 435 436 437 465 466 467 468 469 470 471 496 497 498 499 500 501 502 503 529 530 531 532 533 534 535 536 560 561 562 563 564 565 566 567 568 592 593 594 595 596 597 598 599 600 620 621 622 623 624 625 626 627 628 629 630 631 632 651 652 653 654 655 656 657 658 659 660 661 662 663 679 680 681 682 683 684 685 686 687 688 689 690 703 704 705 706 707 708 709 710 711 712 713 724 725 726 727 728 729 730 741 742 743 )

cmd="source /opt/redmapper/startup.sh && redmapper_run_zred_pixel.py -c /pscratch/sd/m/mkwiecie/redmapper_try1/run/run.yml -p ${pixarr[$SLURM_ARRAY_TASK_ID]} -n 8 -d /pscratch/sd/m/mkwiecie/redmapper_try1/run"

shifter --module=mpich --image docker:ghcr.io/erykoff/redmapper:0.8.5.5 /bin/bash -c "${cmd}"

# background recalibrate
# Edit $HOME/.redmapper_batch.yaml then run below
shifter --module=mpich --image docker:ghcr.io/erykoff/redmapper:0.8.5.5 /bin/bash -c ". /opt/redmapper/startup.sh && /global/homes/m/mkwiecie/des/repos/redmapper/bin/redmapper_batch.py -c run.yml -b NERSC -N 1 -r 1"



# Recalc zred background
shifter --module=mpich --image docker:ghcr.io/erykoff/redmapper:0.8.5.5 /bin/bash -c "source /opt/redmapper/startup.sh && redmapper_make_zred_bkg.py -c run.yml"
