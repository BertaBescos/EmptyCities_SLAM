-- usage example: DATA_ROOT=/path/to/data/ name=expt1 th train.lua 
--
-- code derived from https://github.com/phillipi/pix2pix
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'
require 'cudnn'
require 'criteria'
require 'options'
require 'visualize_save'

---------------------------------------------------------------------------
-- load training options

opt = load_train_options()

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

---------------------------------------------------------------------------

-- create data loader for CARLA images (train and val)
local synth_data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local synth_data = synth_data_loader.new(opt.nThreads, opt)
synth_data_size = synth_data:size()
print("CARLA Dataset Size: ", synth_data_size)
opt.phase = 'val'
local val_synth_data = synth_data_loader.new(opt.nThreads, opt)
print("Validation CARLA Dataset Size: ", val_synth_data:size())

-- create data loader for real images (train and val)
if opt.NSYNTH_DATA_ROOT ~= '' then
	opt.phase = 'train'
	nsynth_data_loader = paths.dofile('data/data_nsynth.lua')
	nsynth_data = nsynth_data_loader.new(opt.nThreads, opt)
	print("Non Synthetic Dataset Size: ", nsynth_data:size())
	opt.phase = 'val'
	val_nsynth_data = nsynth_data_loader.new(opt.nThreads, opt)
	print("Non Synthetic Validation Dataset Size: ", val_nsynth_data:size())
end

opt.phase = 'train'

---------------------------------------------------------------------------

-- set batch/instance normalization
set_normalization(opt.norm)

local real_label = 1
local fake_label = 0
local synth_label = 1

-- load models for generator, discriminator, semantic segmentation, features and SRM noise
load_models()

-- define criteria
if opt.NSYNTH_DATA_ROOT ~= '' then
	criterionSS = WeightedCECriterion()
end

---------------------------------------------------------------------------

-- define helpful variables
local idx_A = {1, opt.input_nc}
local idx_B = {opt.input_nc + 1, opt.input_nc + opt.output_nc}
local idx_C = {opt.input_nc + opt.output_nc + 1, opt.input_nc + opt.output_nc + opt.mask_nc}

optimStateG = {
	learningRate = opt.lr,
	beta1 = opt.beta1,
}
optimStateD = {
	learningRate = opt.lr,
	beta1 = opt.beta1,
}
if opt.NSYTNH_DATA_ROOT ~= '' then
	optimStateSS = {
		learningRate = opt.lr_SS,
		beta1 = opt.beta1_SS,
	}
end

----------------------------------------------------------------------------

realRGB_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSizeH, opt.fineSizeW)
val_realRGB_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSizeH, opt.fineSizeW)
realRGB_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSizeH, opt.fineSizeW)
val_realRGB_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSizeH, opt.fineSizeW)
real_C = torch.Tensor(opt.batchSize, opt.mask_nc, opt.fineSizeH, opt.fineSizeW) --bbescos
val_real_C = torch.Tensor(opt.batchSize, opt.mask_nc, opt.fineSizeH, opt.fineSizeW) --bbescos
fake_B = torch.Tensor(opt.batchSize, opt.output_gan_nc, opt.fineSizeH, opt.fineSizeW)
val_fake_B = torch.Tensor(opt.batchSize, opt.output_gan_nc, opt.fineSizeH, opt.fineSizeW)
real_AC = torch.Tensor(opt.batchSize, opt.input_gan_nc + opt.mask_nc, opt.fineSizeH, opt.fineSizeW)
val_real_AC = torch.Tensor(opt.batchSize, opt.input_gan_nc + opt.mask_nc, opt.fineSizeH, opt.fineSizeW)
real_ABC = torch.Tensor(opt.batchSize, opt.input_gan_nc + opt.output_gan_nc*opt.condition_GAN + opt.mask_nc*opt.condition_mG, opt.fineSizeH, opt.fineSizeW)
val_real_ABC = torch.Tensor(opt.batchSize, opt.input_gan_nc + opt.output_gan_nc*opt.condition_GAN + opt.mask_nc*opt.condition_mG, opt.fineSizeH, opt.fineSizeW)
fake_ABC = torch.Tensor(opt.batchSize, opt.input_gan_nc + opt.output_gan_nc*opt.condition_GAN + opt.mask_nc*opt.condition_mG, opt.fineSizeH, opt.fineSizeW)
val_fake_ABC = torch.Tensor(opt.batchSize, opt.input_gan_nc + opt.output_gan_nc*opt.condition_GAN + opt.mask_nc*opt.condition_mG, opt.fineSizeH, opt.fineSizeW)

epoch_tm = torch.Timer()
tm = torch.Timer()
data_tm = torch.Timer()

----------------------------------------------------------------------------

transfer_to_gpu()

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()
if opt.NSYNTH_DATA_ROOT ~= '' then
	parametersSS, gradParametersSS = netSS:getParameters()
end

load_visualize_options() 

----------------------------------------------------------------------------

function createRealFake()
	-- load real
	data_tm:reset(); data_tm:resume()
	if synth_label == 1 then -- CARLA images
		real_data, data_path = synth_data:getBatch()
	else -- CITYSCAPES images
		real_data, data_path = nsynth_data:getBatch()
	end
	data_tm:stop()

	realRGB_A:copy(real_data[{ {}, idx_A, {}, {} }])
	realRGB_B:copy(real_data[{ {}, idx_B, {}, {} }])
	real_C:copy(real_data[{ {}, idx_C, {}, {} }]) --if CARLA it is dynamic

	-- crete mask
	if synth_label == 0 then
		realBGR_A = realRGB_A:clone():add(1):mul(0.5)
		realBGR_A[1][1] = realRGB_A[1][3]:clone():add(1):mul(0.5)
		realBGR_A[1][3] = realRGB_A[1][1]:clone():add(1):mul(0.5)
		erfnet_C = netSS:forward(realBGR_A) --20 channels
		fake_C = netDynSS:forward(erfnet_C)
	else
		fake_C = real_C:clone()
	end 

	-- convert A and B to gray scale
	if opt.input_gan_nc == 1 then
		realGray_A = util.rgb2gray_batch(realRGB_A)
		realGray_B = util.rgb2gray_batch(realRGB_B)
	else
		realGray_A = realRGB_A
		realGray_B = realRGB_B
	end

	if opt.gpu > 0 then
		realGray_A = realGray_A:cuda()
		realGray_B = realGray_B:cuda()
	end

	-- create fake
	if opt.condition_GAN == 1 then
		real_ABC = torch.cat(realGray_A, realGray_B, 2)
	else
		real_ABC = realGray_B -- unconditional GAN, only penalizes structure in B
	end   

	if opt.condition_mG == 1 then
		real_AC = torch.cat(realGray_A, fake_C, 2)
	else
		real_AC = realGray_A
	end

	fake_B = netG:forward(real_AC)

	if opt.condition_GAN == 1 then
		fake_ABC = torch.cat(realGray_A, fake_B ,2)
	else
		fake_ABC = fake_B -- unconditional GAN, only penalizes structure in B
	end

	if opt.condition_mD == 1 then
		real_ABC = torch.cat(real_ABC, fake_C, 2)
		fake_ABC = torch.cat(fake_ABC,fake_C,2)
	end

	if opt.condition_noise == 1 then
		fake_noise = netNoise:forward(fake_B)
		fake_ABC = torch.cat(fake_ABC, fake_noise, 2)
		real_noise = netNoise:forward(realGray_B)
		real_ABC = torch.cat(real_ABC, real_noise, 2)
	end

	if lossFeatures > 0 then
		if opt.output_gan_nc == 3 then
			temp_realGray_B = netRGB2GrayReal:forward(realGray_B)
			temp_fake_B = netRGB2GrayFake:forward(fake_B)
			feat_real_B = netFeaturesReal:forward(temp_realGray_B)
			feat_fake_B = netFeaturesFake:forward(temp_fake_B)
		else
			feat_real_B = netFeaturesReal:forward(realGray_B)
			feat_fake_B = netFeaturesFake:forward(fake_B)
		end
	end
end

function val_createRealFake()
	 -- load real
	data_tm:reset(); data_tm:resume()
	if synth_label == 1 then -- CARLA images
		val_data, val_data_path = val_synth_data:getBatch()
	else -- CITYSCAPES images
		val_data, val_data_path = val_nsynth_data:getBatch()
	end
	data_tm:stop()

	val_realRGB_A:copy(val_data[{ {}, idx_A, {}, {} }])
	val_realRGB_B:copy(val_data[{ {}, idx_B, {}, {} }])
	val_real_C:copy(val_data[{ {}, idx_C, {}, {} }]) --if CARLA it is dynamic

	-- crete mask
	if synth_label == 0 then
		val_realBGR_A = val_realRGB_A:clone():add(1):mul(0.5)
		val_realBGR_A[1][1] = val_realRGB_A[1][3]:add(1):mul(0.5)
		val_realBGR_A[1][3] = val_realRGB_A[1][1]:add(1):mul(0.5)
		val_erfnet_C = netSS:forward(val_realBGR_A) --20 channels
		val_fake_C = netDynSS:forward(val_erfnet_C)
	else
		val_fake_C = val_real_C:clone()
	end 
	
	-- convert A and B to gray scale
	if opt.input_gan_nc == 1 then
		val_realGray_A = util.rgb2gray_batch(val_realRGB_A)
		val_realGray_B = util.rgb2gray_batch(val_realRGB_B)
	else
		val_realGray_A = val_realRGB_A
		val_realGray_B = val_realRGB_B
	end

	if opt.gpu > 0 then
		val_realGray_A = val_realGray_A:cuda()
		val_realGray_B = val_realGray_B:cuda()
	end

	-- create fake
	if opt.condition_GAN==1 then
		val_real_ABC = torch.cat(val_realGray_A,val_realGray_B,2)
	else
		val_real_ABC = val_realGray_B -- unconditional GAN, only penalizes structure in B
	end   

	if opt.condition_mG == 1 then
		val_real_AC = torch.cat(val_realGray_A, val_fake_C, 2)
	else
		val_real_AC = val_realGray_A
	end

	val_fake_B = netG:forward(val_real_AC)
	
	if opt.condition_GAN==1 then
		val_fake_ABC = torch.cat(val_realGray_A,val_fake_B,2)
	else
		val_fake_ABC = val_fake_B -- unconditional GAN, only penalizes structure in B
	end

	if opt.condition_mD == 1 then
		val_real_ABC = torch.cat(val_real_ABC, val_fake_C, 2)
		val_fake_ABC = torch.cat(val_fake_ABC, val_fake_C, 2)
	end

	if opt.condition_noise == 1 then
		val_fake_noise = netNoise:forward(val_fake_B)
		val_fake_ABC = torch.cat(val_fake_ABC, val_fake_noise, 2)
		val_real_noise = netNoise:forward(val_realGray_B)
		val_real_ABC = torch.cat(val_real_ABC, val_real_noise, 2)
	end

	if lossFeatures > 0 then
		if opt.output_gan_nc == 3 then
			local temp_realGray_B = netRGB2GrayReal:forward(val_realGray_B)
			local temp_fake_B = netRGB2GrayFake:forward(val_fake_B)
			val_feat_real_B = netFeaturesReal:forward(temp_realGray_B)
			val_feat_fake_B = netFeaturesFake:forward(temp_fake_B)
		else
			val_feat_real_B = netFeaturesReal:forward(val_realGray_B)
			val_feat_fake_B = netFeaturesFake:forward(val_fake_B)
		end


	end
end

----------------------------------------------------------------------------
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
	netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
	netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
	 
	gradParametersD:zero()

	-- Real
	local output = netD:forward(real_ABC) -- 1x1x30x30  
	label = torch.FloatTensor(output:size()):fill(real_label)
	if opt.gpu>0 then 
		label = label:cuda()
	end

	if opt.weight == 1 or synth_label == 0 then
		local mask = util.scale_batch(fake_C:clone():float(), output:size(3), output:size(4)):add(1):mul(0.5)
		weightsDiscriminator = torch.zeros(mask:size())
		for i = 1, opt.batchSize do
			local nFeatures = mask[i][mask[i]:gt(0.5)]:numel()
			local nBackground = mask[i][mask[i]:le(0.5)]:numel()
			local valFeatures = mask[i]:numel() / nFeatures
			local valBackground = mask[i]:numel() / nBackground
			if synth_label == 0 then
				weightsDiscriminator[i][mask[i]:le(0.5)] = valBackground
			else
				weightsDiscriminator[i][mask[i]:gt(0.5)] = valFeatures
				weightsDiscriminator[i][mask[i]:le(0.5)] = valBackground
			end
		end

		if opt.gpu > 0 then
			weightsDiscriminator = weightsDiscriminator:cuda()
		end
		criterionDDiscriminator = nn.BCECriterion(weightsDiscriminator)
	else
		criterionDDiscriminator = nn.BCECriterion()
	end

	if opt.gpu > 0 then
		criterionDDiscriminator = criterionDDiscriminator:cuda()
	end

	errD_real = criterionDDiscriminator:forward(output, label)
	df_do = criterionDDiscriminator:backward(output, label) -- 1x1x30x30

	netD:backward(real_ABC, df_do)
	 
	-- Fake
	local output = netD:forward(fake_ABC)
	label:fill(fake_label)

	errD_fake = criterionDDiscriminator:forward(output, label)
	df_do = criterionDDiscriminator:backward(output, label) -- 1x1x30x30 

	netD:backward(fake_ABC, df_do)

	errD = (errD_real + errD_fake)/2
	return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
	netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
	netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
	 
	gradParametersG:zero()
	 
	-- GAN loss
	local df_dg = torch.zeros(fake_B:size())
	if opt.gpu>0 then 
		df_dg = df_dg:cuda()
	end
	 
	output = netD.output -- last call of netD:forward{input_A,input_B} was already executed in fDx, so save computation (with the fake result)
	local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
	if opt.gpu>0 then 
		label = label:cuda();
	end

	if opt.weight == 1 then
		criterionGDiscriminator = nn.WeightedBCECriterion(weightsDiscriminator)
	else
		criterionGDiscriminator = nn.WeightedBCECriterion()
	end
	
	if opt.gpu > 0 then
		criterionGDiscriminator = criterionGDiscriminator:cuda()
	end

	errG = criterionGDiscriminator:forward(output, label)
	df_do = criterionGDiscriminator:backward(output, label)

	df_dg = netD:updateGradInput(fake_ABC, df_do):narrow(2, fake_ABC:size(2) - opt.condition_GAN*opt.output_gan_nc - opt.condition_mD*opt.mask_nc - opt.condition_noise*opt.noise_nc + 1, opt.output_gan_nc)	   

	-- Features loss
	local df_dg_Feat = torch.zeros(fake_B:size())
	if opt.gpu>0 then 
		df_dg_Feat = df_dg_Feat:cuda()
	end
	
	if lossFeatures > 0 then
		if opt.lossDetector == 1 then
			local weightsDetector = computeDetectorWeights(opt.lossDetector, opt.lossOrientation, opt.lossDescriptor, feat_fake_B, feat_real_B)
			if synth_label == 0 then
				local mask = util.scale_batch(fake_C:clone():float(), weightsDetector:size(3), weightsDetector:size(4)):add(1):mul(0.5)
				weightsDetector[mask:gt(0.5)] = 0
			end
			if opt.gpu == 1 then
				weightsDetector = weightsDetector:cuda()
			end
			criterionDetector = nn.WeightedBCECriterion(weightsDetector) --This is the criterion for the features detection
			if opt.gpu > 0 then
				criterionDetector:cuda()
			end
		end

		if opt.lossOrientation == 1 and synth_label == 1 then
			local weightsOrientation = computeOrientationWeights(opt.lossDetector, opt.lossOrientation, opt.lossDescriptor, feat_fake_B, feat_real_B)
			if opt.gpu == 1 then
				weightsOrientation = weightsOrientation:cuda()
			end
			criterionOrientation = nn.WeightedAbsCriterion(weightsOrientation) --This is the criterion for the features orientation
			if opt.gpu > 0 then
				criterionOrientation:cuda()
			end
		end

		if opt.lossDescriptor == 1 and synth_label == 1 then
			local weightsDescriptor = computeDescriptorWeights(opt.lossDetector, opt.lossOrientation, opt.lossDescriptor, feat_fake_B, feat_real_B)
			if opt.gpu == 1 then
				weightsDescriptor = weightsDescriptor:cuda()
			end
			criterionDescriptor = nn.BCECriterion(weightsDescriptor) --This is the criterion for the features detection
			if opt.gpu > 0 then
				criterionDescriptor:cuda()
			end
		end

		errFeatures = 0
		dErrFeatures = torch.Tensor()
		if opt.gpu > 0 then
			dErrFeatures = dErrFeatures:cuda()
		end
		if opt.lossDetector == 1 then
			local errDetector = criterionDetector:forward(feat_fake_B[{{},{1},{},{}}], feat_real_B[{{},{1},{},{}}])
			errFeatures = errFeatures + errDetector
			local dErrDetector = criterionDetector:backward(feat_fake_B[{{},{1},{},{}}], feat_real_B[{{},{1},{},{}}])
			dErrDetector = dErrDetector*opt.lambdaDetector
			dErrFeatures = torch.cat(dErrFeatures, dErrDetector, 2)
		end
		if opt.lossOrientation == 1 and synth_label == 1  then
			local errOrientation = criterionOrientation:forward(feat_fake_B[{{},{1 + opt.lossDetector, 3 + opt.lossDetector},{},{}}], feat_real_B[{{},{1 + opt.lossDetector, 3 + opt.lossDetector},{},{}}])	 
			errFeatures = errFeatures + errOrientation
			local dErrOrientation = criterionOrientation:backward(feat_fake_B[{{},{1 + opt.lossDetector, 3 + opt.lossDetector},{},{}}], feat_real_B[{{},{1 + opt.lossDetector, 3 + opt.lossDetector},{},{}}])
			dErrOrientation = dErrOrientation*opt.lambdaOrientation
			dErrFeatures = torch.cat(dErrFeatures, dErrOrientation, 2)
		end
		if opt.lossDescriptor == 1 and synth_label == 1  then	 
			local errDescriptor = criterionDescriptor:forward(feat_fake_B[{{},{opt.lossDetector + 3*opt.lossOrientation + 1, opt.lossDetector + 3*opt.lossOrientation + 256}, 
				{},{}}], feat_real_B[{{},{opt.lossDetector + 3*opt.lossOrientation + 1, opt.lossDetector + 3*opt.lossOrientation + 256},{},{}}])	 
			errFeatures = errFeatures + errDescriptor
			local dErrDescriptor = criterionDescriptor:backward(feat_fake_B[{{},{opt.lossDetector + 3*opt.lossOrientation + 1, opt.lossDetector + 3*opt.lossOrientation + 256},
				{},{}}], feat_real_B[{{},{opt.lossDetector + 3*opt.lossOrientation + 1, opt.lossDetector + 3*opt.lossOrientation + 256},{},{}}])
			dErrDescriptor = dErrDescriptor*opt.lambdaDescriptor
			dErrFeatures = torch.cat(dErrFeatures, dErrDescriptor, 2)
		end 

		if opt.output_gan_nc == 1 then
			df_dg_Feat = netFeaturesFake:updateGradInput(fake_B, dErrFeatures)
		else
			df_dg_Feat = netFeaturesFake:updateGradInput(temp_fake_B, dErrFeatures)
			df_dg_Feat = netRGB2GrayFake:updateGradInput(fake_B, df_dg_Feat)
		end
	end

	-- Unary loss
	local df_dg_AE = torch.zeros(fake_B:size())
	if opt.gpu>0 then 
		df_dg_AE = df_dg_AE:cuda();
	end

	if opt.weight == 1 or synth_label == 0 then
		local mask = fake_C:clone():float():add(1):mul(0.5)
		local weights = torch.zeros(mask:size())
		local valFeatures = mask:numel() / mask[mask:gt(0.5)]:numel()
		local valBackground = mask:numel() / mask[mask:le(0.5)]:numel()
		if synth_label == 0 then
			weights[mask:le(0.5)] = valBackground
		else
			weights[mask:gt(0.5)] = valFeatures
			weights[mask:le(0.5)] = valBackground
		end
		if opt.output_gan_nc == 3 then
			weights = torch.cat(torch.cat(weights, weights, 2), weights, 2) 
		end
		criterionGenerator = nn.WeightedAbsCriterion(weights) --This is the L1 Loss
	else
		criterionGenerator = nn.AbsCriterion() --This is the L1 Loss
	end

	if opt.gpu>0 then 
		criterionGenerator = criterionGenerator:cuda();
	end

	errL1 = criterionGenerator:forward(fake_B, realGray_B)
	df_dg_AE = criterionGenerator:backward(fake_B, realGray_B)

	--[[print('disc: min', df_dg:min())
	print('disc: max', df_dg:max())
	print('L1: min -> x100', df_dg_AE:min())
	print('L1: max -> x100', df_dg_AE:max())
	print('x10 -> Feat: min', df_dg_Feat:min())
	print('x10 -> Feat: max', df_dg_Feat:max())]]--

	netG:backward(real_AC, df_dg + df_dg_AE:mul(opt.lambda) + df_dg_Feat)   

	return errG, gradParametersG
end

-- create closure to evaluate f(X) and df/dX of ss
local fSSx = function(x)
	gradParametersSS:zero()

	-- GAN loss
	local df_dg = torch.zeros(erfnet_C:size())
	if opt.gpu>0 then 
		df_dg = df_dg:cuda();
	end

	local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for SS cost
	if opt.gpu>0 then 
		label = label:cuda();
	end

	local output = netD.output -- last call of netD:forward{input_A,input_B} was already executed in fDx, so save computation (with the fake result)
	errSS = criterionDDiscriminator:forward(output, label)
	local df_do = criterionDDiscriminator:backward(output, label)

	local df_dp = netD:updateGradInput(fake_ABC, df_do):narrow(2,fake_ABC:size(2) - 
		opt.condition_GAN*opt.output_gan_nc - opt.condition_mD*opt.mask_nc - opt.condition_noise*opt.noise_nc + 1, opt.output_gan_nc)

	local df_dq = netG:updateGradInput(real_AC,df_dp):narrow(2, real_AC:size(2) - 
		opt.mask_nc + 1, opt.mask_nc)

	df_dg = netDynSS:updateGradInput(erfnet_C,df_dq)

	-- SS loss
	local df_dg_SS = torch.zeros(erfnet_C:size())
	if opt.gpu>0 then 
		df_dg_SS = df_dg_SS:cuda();
	end

	fake_C = netSS.output
	errERFNet = criterionSS:forward(erfnet_C, real_C:squeeze(2))
	df_dg_SS = criterionSS:backward(erfnet_C, real_C:squeeze(2))

	netSS:backward(realBGR_A, df_dg + df_dg_SS:mul(opt.lambdaSS))
	
	return errSS, gradParametersSS
end

----------------------------------------------------------------------------
-- train

paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

save_options()

for epoch = opt.epoch_ini, opt.niter do
	epoch_tm:reset()
	for i = 1, math.min(synth_data:size(), opt.ntrain), opt.batchSize do
		tm:reset()
		-- load a batch and run G on that batch
		if opt.NSYNTH_DATA_ROOT ~= '' and epoch > opt.epoch_synth then
			if torch.uniform() > opt.pNonSynth then
				synth_label = 1
			else
				synth_label = 0
			end
		end

		createRealFake()

		-- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
		optim.adam(fDx, parametersD, optimStateD)
		
		-- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
		optim.adam(fGx, parametersG, optimStateG)

		-- (3) Update SS network:
		if synth_label == 0 then optim.adam(fSSx, parametersSS, optimStateSS) end

		opt.counter = opt.counter + 1
		
		display()

		save_display()

		val_display()

		display_plot(epoch, i)

		save_latest_model()

	end

	parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
	parametersG, gradParametersG = nil, nil
	if opt.NSYNTH_DATA_ROOT ~= '' then
		parametersSS, gradParametersSS = nil, nil
	end
	
	print('..........................parameters to nil.......................')

	save_epoch_model(epoch)

	print(('End of epoch %d / %d \t Time Taken: %.3f'):format(epoch, opt.niter, epoch_tm:time().real))
	parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
	parametersG, gradParametersG = netG:getParameters()
	if opt.NSYNTH_DATA_ROOT ~= '' then
		parametersSS, gradParametersSS = netSS:getParameters()
	end
end