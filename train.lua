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

opt = {
	DATA_ROOT = '',				-- path to images (should have subfolders 'train', 'val', etc)
	NSYNTH_DATA_ROOT = '',		-- path to non-synthetic images (should have subfolders 'train', 'val', etc)
	batchSize = 1,          	-- # images in batch
	loadSizeH = 550,			-- scale images height to this size
	loadSizeW = 550,			-- scale images width to this size
	loadSizeH_nsynth = 550,		-- scale non-synthetic images height to this size
	loadSizeW_nsynth = 1100,	-- scale non-synthetic images width to this size
	fineSizeH = 512,			-- then crop to this size
	fineSizeW = 512,			-- then crop to this size
	mask = 1,					-- set to 1 if CARLA images have mask (always on training)
	target = 1,					-- set to 1 if CARLA images have target (always on training)
	ngf = 64,					-- #  of gen filters in first conv layer
	ndf = 64,					-- #  of discrim filters in first conv layer
	input_nc = 3,				-- #  of input image channels (3 if input images are loaded in RGB)
	output_nc = 3,				-- #  of output image channels (3 if target images are loaded in RGB)
	mask_nc = 1,				-- #  of mask channels
	input_gan_nc = 1,			-- #  of input image channels to the GAN architecture (1 if gray-scale)
	output_gan_nc = 1,			-- #  of output image channels from the GAN architecture (1 if gray-scale)
	niter = 200,				-- #  of iter at starting learning rate
	lr = 0.0002,				-- initial learning rate for adam (generator and discriminator)
	beta1 = 0.5,				-- momentum term of adam (generator and discriminator)
	lr_SS = 0.0005,				-- initial learning rate for adam (semantic segmentation)
	beta1_SS = 0.9,				-- momentum term of adam (semantic segmentation)
	ntrain = math.huge,			-- #  of examples per epoch. math.huge for full dataset
	data_aug = 0,				-- data augmentation (if set to 0 not used)
	epoch_synth = 0,			-- train with real and synthetic data from this epoch on
	pNonSynth = 0.10,			-- train with real and synthetic data in this ratio
	display = 1,				-- display samples while training. 0 = false
	display_id = 10,			-- display window id.
	display_plot = 'errERFNet, errFeatures, errL1, val_errL1',	-- which loss values to plot over time.
	gpu = 1,					-- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
	name = 'mGAN',				-- name of the experiment, should generally be passed on the command line
	phase = 'train',			-- train, val, test, etc
	nThreads = 2,				-- # threads for loading data
	val_display_freq = 5000,	-- see validation output every val_display_freq iteration
	save_epoch_freq = 25,		-- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
	save_latest_freq = 5000,	-- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
	print_freq = 50,            -- print the debug information every print_freq iterations
	display_freq = 100,         -- display the current results every display_freq iterations
	save_display_freq = 10000,	-- save the current display of results every save_display_freq_iterations
	continue_train = 0,			-- if continue training, load the latest model: 1: true, 0: false
	epoch_ini = 1,				-- if continue training, at what epoch we start
	counter = 0,				-- it keeps track of iterations
	serial_batches = 0,			-- if 1, takes images in order to make batches, otherwise takes them randomly
	serial_batch_iter = 1,		-- iter into serial image list
	checkpoints_dir = './checkpoints',	-- models are saved here
	ss_dir = './checkpoints/SemSeg/erfnet.net',
	cudnn = 1,					-- set to 0 to not use cudnn
	condition_GAN = 1,          -- set to 0 to use unconditional discriminator
	condition_mG = 1,			-- set to 1 to input also the mask to the generator
	condition_mD = 1,			-- set to 1 to input also the mask to the discriminator
	condition_noise = 1,		-- set to 1 to use SRM noise features for the discriminator
	noise_nc = 3,				-- number of SRM extracted features 
	weight = 1,					-- set to 1 to compensate dynamic/static unbalanced data
	which_model_netD = 'basic',	-- selects model to use for netD
	which_model_netG = 'uresnet_512',	-- selects model to use for netG
	n_layers_D = 0,				-- only used if which_model_netD=='n_layers'
	norm = 'batch',				-- choose either batch or instance normalization
	lambda = 100,				-- weight on L1 term in objective
	lambdaSS = 100,				-- weight on SS term in objective
	lambdaDetector = 10,		-- weight on features detector term in objective
	lambdaOrientation = 0.1,	-- weight on features orientation term in objective
	lambdaDescriptor = 1,		-- weight on features descriptors term in objective
	lossDetector = 1,			-- set to 1 if features detector loss is used
	lossOrientation = 1,		-- set to 1 if features orientation loss is used
	lossDescriptor = 1,			-- set to 1 if features descriptors loss is used
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

-- define helpful variables
local input_nc = opt.input_nc
local output_nc = opt.output_nc
local mask_nc = opt.mask_nc
local input_gan_nc = opt.input_gan_nc
local output_gan_nc = opt.output_gan_nc
local noise_nc = opt.noise_nc

local idx_A = {1, input_nc}
local idx_B = {input_nc + 1, input_nc + output_nc}
local idx_C = {input_nc + output_nc + 1, input_nc + output_nc + mask_nc}

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader for CARLA images (train and val)
local synth_data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local synth_data = synth_data_loader.new(opt.nThreads, opt)
print("CARLA Dataset Size: ", synth_data:size())
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

-- set batch/instance normalization
set_normalization(opt.norm)

local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0
local synth_label = 1

-- function to load discriminator D
function defineD(input_nc, output_nc, ndf)
	local netD = nil
	if opt.condition_GAN==1 or opt.condition_mD==1 then
		input_nc_tmp = input_nc
	else
		input_nc_tmp = 0 -- only penalizes structure in output channels
	end
	if opt.which_model_netD == "basic" then 
		netD = defineD_basic(input_nc_tmp, output_nc, ndf)
	elseif opt.which_model_netD == "n_layers" then 
		netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)  
	else error("unsupported netD model")
	end
	netD:apply(weights_init)
	return netD
end

-- load saved models
if opt.continue_train == 1 then
	print('loading previously trained netG...')
	netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
	print('loading previously trained netD...')
	netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
	if opt.NSYNTH_DATA_ROOT ~= '' then
		if opt.epoch_ini > opt.epoch_synth then
			print('loading previously trained netSS...')
			local ss_path = paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_SS.net')
			netSS = torch.load(ss_path)
			netSS:training()
		else
			print('define model netSS...')
			local ss_path = "./checkpoints/SemSeg/erfnet.net"
			netSS = torch.load(ss_path)
			netSS:training()
		end
	end
else
	print('define model netG...')
	netG = defineG(input_gan_nc + mask_nc*opt.condition_mG, output_gan_nc, ngf)
	print('define model netD...')
	netD = defineD(input_gan_nc + mask_nc*opt.condition_mD + noise_nc*opt.condition_noise, output_gan_nc, ndf)
	if opt.NSYNTH_DATA_ROOT ~= '' then
		print('define model netSS...')
		local ss_path = "./checkpoints/SemSeg/erfnet.net"
		netSS = torch.load(ss_path)
		netSS:training()
	end
end

-- define netDynSS model 
if opt.NSYNTH_DATA_ROOT ~= '' then
	print('define model netDynSS...')
	netDynSS = nn.Sequential()
	local convDyn = nn.SpatialFullConvolution(20,1,1,1,1,1)
	convDyn.weight[{{1,12},1,1,1}] = -8/20 -- Static
	convDyn.weight[{{13,20},1,1,1}] = 12/20 -- Dynamic
	convDyn.bias:zero()
	netDynSS:add(nn.SoftMax())
	netDynSS:add(convDyn)
	netDynSS:add(nn.Tanh())
end

-- define netFeatures model 
local lossFeatures = opt.lossDetector + opt.lossOrientation + opt.lossDescriptor
local stride = 5
if lossFeatures > 0 then
	print('define model netFeatures...')
	netFeaturesReal = define_netFeatures(opt.lossDetector, opt.lossOrientation, opt.lossDescriptor, stride)
	netFeaturesReal:evaluate()
	netFeaturesFake = define_netFeatures(opt.lossDetector, opt.lossOrientation, opt.lossDescriptor, stride)
	netFeaturesFake:evaluate()
	if opt.output_gan_nc == 3 then
		netRGB2GrayReal = define_RGB2Gray()
		netRGB2GrayReal:evaluate()
		netRGB2GrayFake = define_RGB2Gray()
		netRGB2GrayFake:evaluate()
	end
end

if opt.condition_noise == 1 then
	netNoise = define_netNoise(output_gan_nc)
	netNoise:evaluate()
end

-- define criteria
if opt.NSYNTH_DATA_ROOT ~= '' then
	local classes = {'Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence','Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain', 'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle'}
	local classWeights = torch.Tensor(#classes)
	classWeights[1] = 0.0				-- unkown
	classWeights[2] = 2.8149201869965	-- road
	classWeights[3] = 6.9850029945374	-- sidewalk
	classWeights[4] = 3.7890393733978	-- building
	classWeights[5] = 9.9428062438965	-- wall
	classWeights[6] = 9.7702074050903	-- fence
	classWeights[7] = 9.5110931396484	-- pole
	classWeights[8] = 10.311357498169	-- traffic light
	classWeights[9] = 10.026463508606	-- traffic sign
	classWeights[10] = 4.6323022842407	-- vegetation
	classWeights[11] = 9.5608062744141	-- terrain
	classWeights[12] = 7.8698215484619	-- sky
	classWeights[13] = 9.5168733596802	-- person
	classWeights[14] = 10.373730659485	-- rider
	classWeights[15] = 6.6616044044495	-- car
	classWeights[16] = 10.260489463806	-- truck
	classWeights[17] = 10.287888526917	-- bus
	classWeights[18] = 10.289801597595	-- train
	classWeights[19] = 10.405355453491	-- motorcycle
	classWeights[20] = 10.138095855713	-- bicycle
	criterionSS = cudnn.SpatialCrossEntropyCriterion(classWeights)
end

---------------------------------------------------------------------------

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

local realRGB_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSizeH, opt.fineSizeW)
local val_realRGB_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSizeH, opt.fineSizeW)
local realRGB_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSizeH, opt.fineSizeW)
local val_realRGB_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSizeH, opt.fineSizeW)
local real_C = torch.Tensor(opt.batchSize, mask_nc, opt.fineSizeH, opt.fineSizeW) --bbescos
local val_real_C = torch.Tensor(opt.batchSize, mask_nc, opt.fineSizeH, opt.fineSizeW) --bbescos
local fake_B = torch.Tensor(opt.batchSize, output_gan_nc, opt.fineSizeH, opt.fineSizeW)
local val_fake_B = torch.Tensor(opt.batchSize, output_gan_nc, opt.fineSizeH, opt.fineSizeW)
local real_AC = torch.Tensor(opt.batchSize, input_gan_nc + mask_nc, opt.fineSizeH, opt.fineSizeW)
local val_real_AC = torch.Tensor(opt.batchSize, input_gan_nc + mask_nc, opt.fineSizeH, opt.fineSizeW)
local real_ABC = torch.Tensor(opt.batchSize, input_gan_nc + output_gan_nc*opt.condition_GAN + mask_nc*opt.condition_mG, opt.fineSizeH, opt.fineSizeW)
local val_real_ABC = torch.Tensor(opt.batchSize, input_gan_nc + output_gan_nc*opt.condition_GAN + mask_nc*opt.condition_mG, opt.fineSizeH, opt.fineSizeW)
local fake_ABC = torch.Tensor(opt.batchSize, input_gan_nc + output_gan_nc*opt.condition_GAN + mask_nc*opt.condition_mG, opt.fineSizeH, opt.fineSizeW)
local val_fake_ABC = torch.Tensor(opt.batchSize, input_gan_nc + output_gan_nc*opt.condition_GAN + mask_nc*opt.condition_mG, opt.fineSizeH, opt.fineSizeW)

local errD, errG, errL1, errFeatures, errSS, errERFNet = 0, 0, 0, 0, 0, 0
local val_errL1 = 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

----------------------------------------------------------------------------

if opt.gpu > 0 then
	print('transferring to gpu...')
	require 'cunn'
	cutorch.setDevice(opt.gpu)
	realRGB_A = realRGB_A:cuda()
	val_realRGB_A = val_realRGB_A:cuda()
	realRGB_B = realRGB_B:cuda(); fake_B = fake_B:cuda()
	val_realRGB_B = val_realRGB_B:cuda(); val_fake_B = val_fake_B:cuda()
	real_C = real_C:cuda()
	val_real_C = val_real_C:cuda()
	real_ABC = real_ABC:cuda(); fake_ABC = fake_ABC:cuda()
	if opt.cudnn==1 then
		netG = util.cudnn(netG); netD = util.cudnn(netD)
	end
	netD:cuda(); netG:cuda() 
	if lossFeatures > 0 then
		netFeaturesReal:cuda()
		netFeaturesFake:cuda()
		if opt.output_gan_nc == 3 then
			netRGB2GrayReal:cuda()
			netRGB2GrayFake:cuda()
		end
	end
	if opt.NSYNTH_DATA_ROOT ~= '' then
		netDynSS:cuda()
		criterionSS:cuda()
	end
	if opt.condition_noise == 1 then
		netNoise:cuda()
	end
	print('done')
else
	print('running model on CPU')
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.NSYNTH_DATA_ROOT ~= '' then
	parametersSS, gradParametersSS = netSS:getParameters()
end

if opt.display then disp = require 'display' end

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
		criterionGDiscriminator = nn.BCECriterion(weightsDiscriminator)
	else
		criterionGDiscriminator = nn.BCECriterion()
	end
	
	if opt.gpu > 0 then
		criterionGDiscriminator = criterionGDiscriminator:cuda()
	end

	errG = criterionGDiscriminator:forward(output, label)
	df_do = criterionGDiscriminator:backward(output, label)

	df_dg = netD:updateGradInput(fake_ABC, df_do):narrow(2, fake_ABC:size(2) - opt.condition_GAN*output_gan_nc - opt.condition_mD*mask_nc - opt.condition_noise*noise_nc + 1, output_gan_nc)	   

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
			criterionDetector = nn.BCECriterion(weightsDetector) --This is the criterion for the features detection
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
		if output_gan_nc == 3 then
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
		opt.condition_GAN*output_gan_nc - opt.condition_mD*mask_nc - opt.condition_noise*noise_nc + 1, output_gan_nc)

	local df_dq = netG:updateGradInput(real_AC,df_dp):narrow(2, real_AC:size(2) - 
		mask_nc + 1, mask_nc)

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
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

-- parse diplay_plot string into table
opt.display_plot = string.split(string.gsub(opt.display_plot, "%s+", ""), ",")
for k, v in ipairs(opt.display_plot) do
	 if not util.containsValue({"errG", "errD", "errL1", "errFeatures","errERFNet", "val_errG", "val_errD", "val_errL1", "val_errFeatures"}, v) then 
		  error(string.format('bad display_plot value "%s"', v)) 
	 end
end

-- display plot config
local plot_config = {
  title = "Loss over time",
  labels = {"epoch", unpack(opt.display_plot)},
  ylabel = "loss",
}

-- display plot vars
local plot_data = {}
local plot_win
local aspect_ratio = opt.fineSizeW / opt.fineSizeH

----------------------------------------------------------------------------

-- main loop
local counter = opt.counter -- 0
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

		-- display
		counter = counter + 1
		if counter % opt.display_freq == 0 and opt.display then	
			createRealFake()
			local img_input = util.scale_batch(realGray_A:float(),100,100*aspect_ratio):add(1):div(2)
			if input_gan_nc == 3 then
				img_input = util.deprocess_batch(img_input)
			end
			disp.image(img_input, {win=opt.display_id, title=opt.name .. ' input'})
			local mask_input = util.scale_batch(real_C:float(),100,100*aspect_ratio):add(1):div(2)
			disp.image(mask_input, {win=opt.display_id+1, title=opt.name .. ' mask'})
			local img_output = util.scale_batch(fake_B:float(),100,100*aspect_ratio):add(1):div(2)
			if output_gan_nc == 3 then
				img_input = util.deprocess_batch(img_output)
			end
			disp.image(img_output, {win=opt.display_id+2, title=opt.name .. ' output'})
			local img_target = util.scale_batch(realGray_B:float(),100,100*aspect_ratio):add(1):div(2)
			if output_gan_nc == 3 then
				img_target = util.deprocess_batch(img_target)
			end
			disp.image(img_target, {win=opt.display_id+3, title=opt.name .. ' target'})
			if opt.lossDetector == 1 then
				local output_map = feat_fake_B[{{},{1},{},{}}]:clone()
				output_map[output_map:gt(0.5)] = 1
				output_map[output_map:le(0.5)] = 0
				local img_output_features = util.scale_batch(output_map:float(),100,100*aspect_ratio)
				disp.image(img_output_features, {win=opt.display_id+4, title=opt.name .. ' output_features'})
				local target_map = feat_real_B[{{},{1},{},{}}]:clone()
				target_map[target_map:gt(0.5)] = 1
				target_map[target_map:le(0.5)] = 0
				local img_target_features = util.scale_batch(target_map:float(),100,100*aspect_ratio)
				disp.image(img_target_features, {win=opt.display_id+5, title=opt.name .. ' label_features'})
			end
			if opt.lossOrientation == 1 then
				local img_output_features = util.scale_batch(feat_fake_B[{{},{opt.lossDetector + 1},{},{}}]:float(),100,100*aspect_ratio)
				disp.image(img_output_features, {win=opt.display_id+6, title=opt.name .. ' output_m10'})
				local img_target_features = util.scale_batch(feat_real_B[{{},{opt.lossDetector + 1},{},{}}]:float(),100,100*aspect_ratio)
				disp.image(img_target_features, {win=opt.display_id+7, title=opt.name .. ' label_m10'})
				local img_output_features = util.scale_batch(feat_fake_B[{{},{opt.lossDetector + 2},{},{}}]:float(),100,100*aspect_ratio)
				disp.image(img_output_features, {win=opt.display_id+8, title=opt.name .. ' output_m00'})
				local img_target_features = util.scale_batch(feat_real_B[{{},{opt.lossDetector + 2},{},{}}]:float(),100,100*aspect_ratio)
				disp.image(img_target_features, {win=opt.display_id+9, title=opt.name .. ' label_m00'})
				local img_output_features = util.scale_batch(feat_fake_B[{{},{opt.lossDetector + 3},{},{}}]:float(),100,100*aspect_ratio)
				disp.image(img_output_features, {win=opt.display_id+10, title=opt.name .. ' output_m01'})
				local img_target_features = util.scale_batch(feat_real_B[{{},{opt.lossDetector + 3},{},{}}]:float(),100,100*aspect_ratio)
				disp.image(img_target_features, {win=opt.display_id+11, title=opt.name .. ' label_m01'})
			end
			if opt.lossDescriptor == 1 then
				local img_output_features = util.scale_batch(feat_fake_B[{{},{opt.lossDetector + 3*opt.lossOrientation + 1},{},{}}]:float(),100,100*aspect_ratio)
				disp.image(img_output_features, {win=opt.display_id+12, title=opt.name .. ' output_pair1'})
				local img_target_features = util.scale_batch(feat_real_B[{{},{opt.lossDetector + 3*opt.lossOrientation + 1},{},{}}]:float(),100,100*aspect_ratio)
				disp.image(img_target_features, {win=opt.display_id+13, title=opt.name .. ' label_pair1'})
				local img_output_features = util.scale_batch(feat_fake_B[{{},{opt.lossDetector + 3*opt.lossOrientation + 2},{},{}}]:float(),100,100*aspect_ratio)
				disp.image(img_output_features, {win=opt.display_id+14, title=opt.name .. ' output_pair2'})
				local img_target_features = util.scale_batch(feat_real_B[{{},{opt.lossDetector + 3*opt.lossOrientation + 2},{},{}}]:float(),100,100*aspect_ratio)
				disp.image(img_target_features, {win=opt.display_id+15, title=opt.name .. ' label_pair2'})
			end
			if synth_label == 0 then
				local dyn_mask_output = util.scale_batch(fake_C:float(),100,100*aspect_ratio):add(1):div(2)
				disp.image(dyn_mask_output, {win=opt.display_id+17, title=opt.name .. ' dynamic_mask'})
			end
		end

		-- write display visualization to disk
		-- runs on the first batchSize images in the opt.phase set
		if counter % opt.save_display_freq == 0 and opt.display then
			local serial_batches=opt.serial_batches
			opt.serial_batches=1
			opt.serial_batch_iter=1
			
			local image_out = nil
			local N_save_display = 10 
			local N_save_iter = torch.max(torch.Tensor({1, torch.floor(N_save_display/opt.batchSize)}))
			for i3=1, N_save_iter do
				createRealFake()
				print('save to the disk')
				for i2=1, fake_B:size(1) do
					if image_out==nil then
						if input_gan_nc == 1 then 
							image_out = torch.cat(realGray_A[i2]:float():add(1):div(2), fake_B[i2]:float():add(1):div(2), 3)
						else
							image_out = torch.cat(util.deprocess(realGray_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3)
						end
					else
						if input_gan_nc == 1 then
							image_out = torch.cat(image_out, torch.cat(realGray_A[i2]:float():add(1):div(2), fake_B[i2]:float():add(1):div(2),3), 2)
						else
							image_out = torch.cat(image_out, torch.cat(util.deprocess(realGray_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3), 2)
						end
					end
				end
			end
			image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
			opt.serial_batches=serial_batches
		end
		
		-- validation display
		if (counter % opt.val_display_freq == 0 or counter == 1) and opt.display then
			val_createRealFake()
			val_errL1 = criterionGenerator:forward(val_fake_B, val_realGray_B)
			local img_input = util.scale_batch(val_realGray_A:float(),100,100*aspect_ratio):add(1):div(2)
			if input_gan_nc == 3 then
				img_input = util.deprocess_batch(img_input)
			end
			disp.image(img_input, {win=opt.display_id+20, title=opt.name .. ' val_input'})
			local mask_input = util.scale_batch(val_real_C:float(),100,100*aspect_ratio):add(1):div(2)
			disp.image(mask_input, {win=opt.display_id+21, title=opt.name .. ' val_mask'})
			local img_output = util.scale_batch(val_fake_B:float(),100,100*aspect_ratio):add(1):div(2)
			if output_gan_nc == 3 then
				img_output = util.deprocess_batch(img_output)
			end
			disp.image(img_output, {win=opt.display_id+22, title=opt.name .. ' val_output'})
			local img_target = util.scale_batch(val_realGray_B:float(),100,100*aspect_ratio):add(1):div(2)
			if output_gan_nc == 3 then
				img_target = util.deprocess_batch(img_target)
			end
			disp.image(img_target, {win=opt.display_id+23, title=opt.name .. ' val_target'})
			if opt.lossDetector == 1 then
				local val_output_map = val_feat_fake_B[{{},{1},{},{}}]:clone()
				val_output_map[val_output_map:gt(0.5)] = 1
				val_output_map[val_output_map:le(0.5)] = 0
				local img_output_features = util.scale_batch(val_output_map:float(),100,100)
				disp.image(img_output_features, {win=opt.display_id+24, title=opt.name .. ' val_output_features'})
				local val_target_map = val_feat_real_B[{{},{1},{},{}}]:clone()
				val_target_map[val_target_map:gt(0.5)] = 1
				val_target_map[val_target_map:le(0.5)] = 0
				local img_target_features = util.scale_batch(val_target_map:float(),100,100)
				disp.image(img_target_features, {win=opt.display_id+25, title=opt.name .. ' val_target_features'})
			end
			if synth_label == 0 then
				local dyn_mask_output = util.scale_batch(val_fake_C:float(),100,100*aspect_ratio):add(1):div(2)
				disp.image(dyn_mask_output, {win=opt.display_id+27, title=opt.name .. ' val_dynamic_mask'})
			end
		end

		-- logging and display plot
		if counter % opt.print_freq == 0 then
			local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errL1=errL1 and errL1 or -1, errFeatures=errFeatures and errFeatures or -1, errERFNet=errERFNet and errERFNet or -1, val_errL1=val_errL1 and val_errL1 or -1}
			local curItInBatch = ((i-1) / opt.batchSize)
			local totalItInBatch = math.floor(math.min(synth_data:size(), opt.ntrain) / opt.batchSize)
			print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
					.. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f ErrFeatures: %.4f'):format(
					 epoch, curItInBatch, totalItInBatch,
					 tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
					 errG, errD, errL1, errFeatures))
			local plot_vals = { epoch + curItInBatch / totalItInBatch }
			for k, v in ipairs(opt.display_plot) do
				if loss[v] ~= nil then
				   plot_vals[#plot_vals + 1] = loss[v]
				end
			end
			
			-- update display plot
			if opt.display then
				table.insert(plot_data, plot_vals)
				plot_config.win = plot_win
				plot_win = disp.plot(plot_data, plot_config)
			end
		end

		-- save latest model
		if counter % opt.save_latest_freq == 0 then
			print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
			torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
			torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
			if opt.NSYNTH_DATA_ROOT ~= '' then
				torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_SS.net'), netSS:clearState())
			end
		end
	end

	parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
	parametersG, gradParametersG = nil, nil
	if opt.NSYNTH_DATA_ROOT ~= '' then
		parametersSS, gradParametersSS = nil, nil
	end
	
	print('..........................parameters to nil.......................')

	if epoch % opt.save_epoch_freq == 0 then
		torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
		torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
		if opt.NSYNTH_DATA_ROOT ~= '' then
			torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_SS.net'), netSS:clearState())
		end
	end

	print(('End of epoch %d / %d \t Time Taken: %.3f'):format(epoch, opt.niter, epoch_tm:time().real))
	parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
	parametersG, gradParametersG = netG:getParameters()
	if opt.NSYNTH_DATA_ROOT ~= '' then
		parametersSS, gradParametersSS = netSS:getParameters()
	end
end