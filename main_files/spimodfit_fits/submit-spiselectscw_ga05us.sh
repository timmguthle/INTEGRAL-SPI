#!/bin/sh

CMD_NAME=spiselectscw

CMD=/afs/ipp-garching.mpg.de/mpe/gamma/instruments/integral/software/local/spiselectscw/4.02/amd64_sles11_g++/spiselectscw


#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# please modify the email address
EMAIL=intopt@mpe.mpg.de

if [ $# -ne 1 ] ; then
	echo Usage:
	echo "     $0 run_ID"
	exit
fi

RUN_ID=$1
PARFILE=spiselectscw.${RUN_ID}.par
LOGFILE=spiselectscw.log

if [ ! -f $PARFILE ] ; then
    echo "Parameter file $PARFILE doesn't exist. Exit!"
    echo
    exit
fi

export CFITSIO_INCLUDE_FILES=/afs/ipp/mpe/gamma/instruments/integral/software/osa/osa-10.0/linux64_sw-10.0/templates
export PFILES=.

if [ ! -d ${RUN_ID} ] ; then
    mkdir ${RUN_ID}
fi
cd ${RUN_ID}
if [ ! -d spi ]; then
    mkdir spi
fi

if [ ! -f spi_off_det.fits ] ; then
    ln -sf /afs/ipp/mpe/gamma/instruments/integral/software/local/spiselectscw/current/spi_off_det.fits
fi
if [ ! -f spi_gnrl_bti.fits ] ; then
    ln -sf /afs/ipp/mpe/gamma/instruments/integral/data/ic/spi/lim/spi_gnrl_bti_0005.fits spi_gnrl_bti.fits
fi
#ln -sf ../$PARFILE spiselectscw.par
cp -pf ../$PARFILE spiselectscw.par

create_scw_file=`grep create_scw_file spiselectscw.par | cut -d',' -f4`
scw_file=`grep ^scw_file spiselectscw.par | cut -d'"' -f2`
if [ ${create_scw_file} = 0 -a ! -f ${scw_file} ] ; then
    ln -s ../scw.fits
fi

date > $LOGFILE
$CMD >> $LOGFILE 2>&1

status=$?
date >> $LOGFILE

#echo "spiselectscw (${RUN_ID}) exit status: $status" | mail -s "Job finish. $PWD" $EMAIL

exit
