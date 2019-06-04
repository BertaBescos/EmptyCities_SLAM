
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
print(os.getenv('DATA_ROOT'))
opt.data = paths.concat(os.getenv('DATA_ROOT'), opt.phase)

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
local loadSizeH   = {input_nc, opt.loadSizeH}
local loadSizeW   = {input_nc, opt.loadSizeW}
local sampleSizeH = {input_nc, opt.fineSizeH}
local sampleSizeW = {input_nc, opt.fineSizeW}

local preprocess = function(imA, imB, imC)

	if imB == nil then
		if imC == nil then
			if opt.data_aug == 1 then imA,imB,imC = data_aug.apply(imA,imA,imA) else imB, imC = imA, imA end
		else
			if opt.data_aug == 1 then imA,imB,imC = data_aug.apply(imA,imA,imC) else imB = imA end
		end
	else
		if imC == nil then
			if opt.data_aug == 1 then imA,imB,imC = data_aug.apply(imA,imB,imA) else imC = imA end
		else
			if opt.data_aug == 1 then imA,imB,imC = data_aug.apply(imA,imB,imC) end
		end
	end

	imA = image.scale(imA, loadSizeW[2], loadSizeH[2])
	imB = image.scale(imB, loadSizeW[2], loadSizeH[2])
	imC = image.scale(imC, loadSizeW[2], loadSizeH[2]) -- between 0 and 1
	imC[imC:gt(0)] = 1

	local perm = torch.LongTensor{3, 2, 1}
	imA = imA:index(1, perm)--:mul(256.0): brg, rgb
	imB = imB:index(1, perm)
	
	local oW = sampleSizeW[2]
	local oH = sampleSizeH[2]
	local iH = imA:size(2)
	local iW = imA:size(3)

	if iH~=oH then     
		h1 = math.ceil(torch.uniform(1e-2, iH-oH))
		--h1 = 8
		--h1 = 16
	end
	
	if iW~=oW then
		w1 = math.ceil(torch.uniform(1e-2, iW-oW))
		--w1 = 314
		--w1 = 16
	end
	if iH ~= oH or iW ~= oW then 
		imA = image.crop(imA, w1, h1, w1 + oW, h1 + oH)
		imB = image.crop(imB, w1, h1, w1 + oW, h1 + oH)
		imC = image.crop(imC, w1, h1, w1 + oW, h1 + oH)
	end

	imA = imA:mul(2):add(-1)
	imB = imB:mul(2):add(-1)
	imC = imC:mul(2):add(-1) -- min(imC) = -1 & max(imC) = 1
	assert(imA:max()<=1,"A: badly scaled inputs")
	assert(imA:min()>=-1,"A: badly scaled inputs")
	assert(imB:max()<=1,"B: badly scaled inputs")
	assert(imB:min()>=-1,"B: badly scaled inputs")
	assert(imC:max()<=1,"C: badly scaled inputs")
	assert(imC:min()>=-1,"C: badly scaled inputs")
	
	if opt.target == '' then
		imB = nil
	end
	if opt.mask == '' then
		imC = nil
	end

	return imA, imB, imC
end

--local function to load the images
local function loadImage(path)
	local input = image.load(path, input_nc, 'float')
	local h = input:size(2)
	local w = input:size(3)

	imA, imB, imC = nil

	if opt.mask == '' and opt.target == '' then
		imA = input
	end 
	if opt.mask == '' and opt.target ~= '' then
		imA = image.crop(input, 0, 0, w/2, h)
		imB = image.crop(input, w/2, 0, w, h)
	end
	if opt.mask ~= '' and opt.target == '' then
		imA = image.crop(input, 0, 0, w/2, h)
		imC = image.crop(input, w/2, 0, w, h)
	end
	if opt.mask ~= '' and opt.target ~= '' then
		imA = image.crop(input, 0, 0, w/3, h)
		imB = image.crop(input, w/3, 0, 2*w/3, h)
		imC = image.crop(input, 2*w/3, 0, w, h)
	end
	
	return imA, imB, imC
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
	collectgarbage()
	local imA, imB, imC = loadImage(path)
	imA, imB, imC = preprocess(imA, imB, imC)

	if imB ~= nil and imC ~= nil then
		im = torch.cat(imA, imB, 1)
		im = torch.cat(im, imC, 1)
	end
	if imB == nil and imC ~= nil then
		im = torch.cat(imA, imC, 1)
	end
	if imB ~= nil and imC == nil then
		im = torch.cat(imA, imB, 1)
	end
	if imB == nil and imC == nil then
		im = imA
	end

	return im
end

--------------------------------------

-- trainLoader
print('trainCache', trainCache)
print('Creating train metadata')
print('serial batch:, ', opt.serial_batches)

nc = input_nc
if opt.target ~= '' then nc = nc + output_nc end
if opt.mask ~= '' then nc = nc + 3 end

trainLoader = dataLoader{
	paths = {opt.data},
	loadSize = {input_nc, loadSizeH[2], loadSizeW[2]},
	sampleSize = {nc, sampleSizeH[2], sampleSizeW[2]},
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
