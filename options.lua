
function load_train_options()
	local opt = {
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

	return opt
end

function load_visualize_options() 

	if opt.display then disp = require 'display' end

	-- parse diplay_plot string into table
	opt.display_plot = string.split(string.gsub(opt.display_plot, "%s+", ""), ",")
	for k, v in ipairs(opt.display_plot) do
		 if not util.containsValue({"errG", "errD", "errL1", "errFeatures","errERFNet", "val_errG", "val_errD", "val_errL1", "val_errFeatures"}, v) then 
			  error(string.format('bad display_plot value "%s"', v)) 
		 end
	end

	-- display plot config
	plot_config = {
	  title = "Loss over time",
	  labels = {"epoch", unpack(opt.display_plot)},
	  ylabel = "loss",
	}

	-- display plot vars
	plot_data = {}
	local plot_win
	aspect_ratio = opt.fineSizeW / opt.fineSizeH

	errD, errG, errL1, errFeatures, errSS, errERFNet = 0, 0, 0, 0, 0, 0
	val_errL1 = 0
end

function load_test_options()
	local opt = {
		DATA_ROOT = '',				-- path to images (should have subfolders 'train', 'val', etc)
		input = '',					-- path to input image
		mask = '',					-- path to mask input image
		output =  '',				-- path to save output image
		target = '',				-- path to objective image
		mask_output = 'mask_output.png',	-- path to mask output image
		data_aug = 0,				-- data augmentation (if set to 0 not used)
		batchSize = 1,				-- # images in batch
		loadSizeH = 550,			-- scale images height to this size
		loadSizeW = 550,			-- scale images width to this size
		fineSizeH = 512,			-- then crop to this size
		fineSizeW = 512,			-- then crop to this size
		display = 1,				-- display samples while testing. 0 = false
		display_id = 200,			-- display window id.
		gpu = 1,					-- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
		phase = 'test',				-- train, val, test ,etc
		aspect_ratio = 1.0,			-- aspect ratio of result images
		name = 'mGAN',				-- name of experiment, selects which model to run, should generally should be passed on command line
		input_nc = 3,               -- #  of input image channels (3 if input images are loaded in RGB)
		output_nc = 3,              -- #  of output image channels (3 if target images are loaded in RGB)
		input_gan_nc = 1,			-- #  of input image channels to the GAN architecture (1 if gray-scale)
		output_gan_nc = 1;			-- #  of output image channels to the GAN architecture (1 if gray-scale)
		mask_nc = 1,				-- #  of mask channels
		serial_batches = 1,			-- if 1, takes images in order to make batches, otherwise takes them randomly
		serial_batch_iter = 1,		-- iter into serial image list
		cudnn = 1,					-- set to 0 to not use cudnn (untested)
		checkpoints_dir = './checkpoints',	-- loads models from here
		results_dir='./results/',	-- saves results here
		which_epoch = 'latest',		-- which epoch to test? set to 'latest' to use latest cached model
		condition_mG = 1,
		netSS_name = 'SemSeg/erfnet.net'
	}

	return opt
end
