
--[[
		This data loader is a modified version of the one from dcgan.torch
		(see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).
		Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
		Copyright (c) 2015-present, Facebook, Inc.
		All rights reserved.
		This source code is licensed under the BSD-style license found in the
		LICENSE file in the root directory of this source tree. An additional grant
		of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset.lua')
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
print(os.getenv('NSYNTH_DATA_ROOT'))
opt.data = paths.concat(os.getenv('NSYNTH_DATA_ROOT'), opt.phase)

-- This file contains the data augmentation techniques.
paths.dofile('data_aug.lua')

if not paths.dirp(opt.data) then
	error('Did not find directory: ' .. opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local input_nc = opt.input_nc -- input channels
local output_nc = opt.output_nc
local mask_nc = opt.mask_nc
local loadSizeH   = {input_nc, opt.loadSizeH_nsynth}
local loadSizeW   = {input_nc, opt.loadSizeW_nsynth}
local sampleSizeH = {input_nc, opt.fineSizeH}
local sampleSizeW = {input_nc, opt.fineSizeW}

local preprocess = function(imA, imB, imC)

	if opt.data_aug == 1 then
		imA,imB,imC = data_aug.apply(imA,imB,imC)
	end

	imA = image.scale(imA, loadSizeW[2], loadSizeH[2])
	imB = image.scale(imB, loadSizeW[2], loadSizeH[2])
	imC = image.scale(imC, loadSizeW[2], loadSizeH[2]) -- between 0 and 1

	local perm = torch.LongTensor{3, 2, 1}
	imA = imA:index(1, perm)--:mul(256.0): brg, rgb
	imB = imB:index(1, perm)
	
	local oW = sampleSizeW[2]
	local oH = sampleSizeH[2]
	local iH = imA:size(2)
	local iW = imA:size(3)
	
	if iH~=oH then     
		--h1 = 8
		h1 = math.ceil(torch.uniform(1e-2, iH-oH))
	end 
	
	if iW~=oW then
		--w1 = 314
		w1 = math.ceil(torch.uniform(1e-2, iW-oW))
	end
	if iH ~= oH or iW ~= oW then 
		imA = image.crop(imA, w1, h1, w1 + oW, h1 + oH)
		imB = image.crop(imB, w1, h1, w1 + oW, h1 + oH)
		imC = image.crop(imC, w1, h1, w1 + oW, h1 + oH)
	end

	imA = imA:mul(2):add(-1)
	imB = imB:mul(2):add(-1)
	assert(imA:max()<=1,"A: badly scaled inputs")
	assert(imA:min()>=-1,"A: badly scaled inputs")
	assert(imB:max()<=1,"B: badly scaled inputs")
	assert(imB:min()>=-1,"B: badly scaled inputs")

	return imA, imB, imC
end

--local function loadImage
local function loadImage(path)
	 local input = image.load(path, 3, 'float')
	 local input_mask = image.load(path, 3, 'byte')	

	 local h = input:size(2)
	 local w = input:size(3)

	 local imA = image.crop(input, 0, 0, w/2, h)
	 local imB = imA:clone()
	 local imC = image.crop(input_mask, w/2, 0, w, h)

	 imC = imC[1]:resize(1,imC:size(2),imC:size(3)):float() -- 1 channel

	 classMap = {  [-1] =  {1}, -- licence plate
								[0]  =  {1}, -- Unlabeled
								[1]  =  {1}, -- Ego vehicle
								[2]  =  {1}, -- Rectification border
								[3]  =  {1}, -- Out of roi
								[4]  =  {1}, -- Static
								[5]  =  {1}, -- Dynamic
								[6]  =  {1}, -- Ground
								[7]  =  {2}, -- Road
								[8]  =  {3}, -- Sidewalk
								[9]  =  {1}, -- Parking
								[10] =  {1}, -- Rail track
								[11] =  {4}, -- Building
								[12] =  {5}, -- Wall
								[13] =  {6}, -- Fence
								[14] =  {1}, -- Guard rail
								[15] =  {1}, -- Bridge
								[16] =  {1}, -- Tunnel
								[17] =  {7}, -- Pole
								[18] =  {1},  -- Polegroup
								[19] =  {8}, -- Traffic light
								[20] =  {9}, -- Traffic sign
								[21] = {10}, -- Vegetation
								[22] = {11}, -- Terrain
								[23] = {12}, -- Sky
								[24] = {13}, -- Person
								[25] = {14}, -- Rider
								[26] = {15}, -- Car
								[27] = {16}, -- Truck
								[28] = {17}, -- Bus
								[29] =  {1}, -- Caravan
								[30] =  {1}, -- Trailer
								[31] = {18}, -- Train
								[32] = {19}, -- Motorcycle
								[33] = {20}, -- Bicycle
								}

	 imC:apply(function(x) return classMap[x][1] end)

	 return imA, imB, imC
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
	collectgarbage()
	local imA, imB, imC = loadImage(path)
	imA, imB, imC = preprocess(imA, imB, imC)
	imAB = torch.cat(imA, imB, 1)
	imABC = torch.cat(imAB, imC, 1)
	return imABC
end

--------------------------------------
-- trainLoader
print('trainCache', trainCache)
print('Creating train metadata')
print('serial batch:, ', opt.serial_batches)
trainLoader = dataLoader{
		paths = {opt.data},
		loadSize = {input_nc, loadSizeH[2], loadSizeW[2]},
		sampleSize = {input_nc+output_nc+mask_nc, sampleSizeH[2], sampleSizeW[2]},
		split = 100,
		serial_batches = opt.serial_batches, 
		verbose = true
 }
trainLoader.sampleHookTrain = trainHook

collectgarbage()

-- do some sanity checks on trainLoader
do
	 local class = trainLoader.imageClass
	 local nClasses = #trainLoader.classes
	 assert(class:max() <= nClasses, "class logic has error")
	 assert(class:min() >= 1, "class logic has error")
end
