cp ./Data/u018_w001/u018_w001_accelerometer.log ./Data_test/u000_w000/u000_w000_accelerometer.log
oldnum=`cut -d ',' -f2 ./Data_test/TESTRUNS`
newnum=`expr $oldnum + 1`
sed -i "s/$oldnum\$/$newnum/g" ./Data_test/TESTRUNS
