for i in 0.1  0.2  0.3  0.4  0.5  3.9  3.91  5.9  5.91  8.12  8.14  pointnet2_charlesmsg_0  pointnet2_charlesmsg_1  pointnet2_charlesssg_1  pointnet2_meteornet_1
do
 echo $i
 mkdir -p $i/train
 mkdir -p $i/unseen
 mkdir -p $i/seen
done
