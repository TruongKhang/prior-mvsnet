#scp -P 10022 khang@143.248.135.115:~/project/cascade-stereo/CasMVSNet/outputs/*.ply outputs/
#matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/code/cascade-stereo/CasMVSNet/evaluations/dtu/BaseEvalMain_web.m');exit;" | tail -n +11
#matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/code/cascade-stereo/CasMVSNet/evaluations/dtu/ComputeStat_web.m');exit;" | tail -n +11
#mv outputs/eval_out/TotalStat_mvsnet_Eval_.txt results/eval_gipuma.txt
#rm -rf outputs/eval_out
#rm outputs/*.ply
#rm -rf outputs/scan*/points_mvsnet/con*
#####################################
./test.sh model_000022.ckpt --outdir outputs --interval_scale 1.06 --filter_method gipuma --prob_threshold 0.8 --disp_threshold 0.25 --num_consistent 4
python move.py outputs
matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/code/cascade-stereo/CasMVSNet/evaluations/dtu/BaseEvalMain_web.m');exit;" | tail -n +11
matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/code/cascade-stereo/CasMVSNet/evaluations/dtu/ComputeStat_web.m');exit;" | tail -n +11
mv outputs/eval_out/TotalStat_mvsnet_Eval_.txt results/eval_0.txt
rm -rf outputs/eval_out
rm outputs/*.ply
rm -rf outputs/scan*/points_mvsnet/con*
#scp -P 10022 khang@143.248.135.115:~/project/cascade-stereo/CasMVSNet/outputs/*.ply outputs/
./test.sh model_000022.ckpt --outdir outputs --interval_scale 1.06 --filter_method gipuma --prob_threshold 0.8 --disp_threshold 0.2 --num_consistent 3
python move.py outputs
matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/code/cascade-stereo/CasMVSNet/evaluations/dtu/BaseEvalMain_web.m');exit;" | tail -n +11
matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/code/cascade-stereo/CasMVSNet/evaluations/dtu/ComputeStat_web.m');exit;" | tail -n +11
mv outputs/eval_out/TotalStat_mvsnet_Eval_.txt results/eval_1.txt
rm -rf outputs/eval_out
rm outputs/*.ply
rm -rf outputs/scan*/points_mvsnet/con*
######
# cp outputs_copy/*.ply outputs/
#scp -P 10022 khang@143.248.135.115:~/project/cascade-stereo/CasMVSNet/outputs/*.ply outputs/
#./test.sh model_000022.ckpt --outdir outputs --interval_scale 1.06 --filter_method gipuma --prob_threshold 0.8 --disp_threshold 0.1 --num_consistent 1
#python move.py outputs
#matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/code/cascade-stereo/CasMVSNet/evaluations/dtu/BaseEvalMain_web.m');exit;" | tail -n +11
#matlab -nodisplay -nosplash -nodesktop -r "run('/home/khangtg/Documents/lab/code/cascade-stereo/CasMVSNet/evaluations/dtu/ComputeStat_web.m');exit;" | tail -n +11
#mv outputs/eval_out/TotalStat_mvsnet_Eval_.txt results/eval_8.txt
#rm -rf outputs/eval_out
#rm outputs/*.ply
#rm -rf outputs/scan*/points_mvsnet/con*
######
