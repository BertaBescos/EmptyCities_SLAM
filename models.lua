require 'nngraph'

-- definition of normalization types
normalization = nil
function set_normalization(norm)
	if norm == 'instance' then
		require 'util.InstanceNormalization'
		print('use InstanceNormalization')
		normalization = nn.InstanceNormalization
	elseif norm == 'batch' then
		print('use SpatialBatchNormalization')
		normalization = nn.SpatialBatchNormalization
	end
end

-- initialization of model weights
function weights_init(m)
	local name = torch.type(m)
	if name:find('Convolution') then
		m.weight:normal(0.0, 0.02)
		m.bias:fill(0)
	elseif name:find('BatchNormalization') then
		if m.weight then m.weight:normal(1.0, 0.02) end
		if m.bias then m.bias:fill(0) end
	end
end

-- function to load generator G
function defineG(input_nc, output_nc, ngf)
	local netG = nil
	if     opt.which_model_netG == "encoder_decoder" then 
		netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
	elseif opt.which_model_netG == "unet" then 
		netG = defineG_unet(input_nc, output_nc, ngf)
	elseif opt.which_model_netG == "unet_128" then 
		netG = defineG_unet_128(input_nc, output_nc, ngf)
	elseif opt.which_model_netG == "unet_upsample" then
		netG = defineG_unet_upsampling(input_nc, output_nc, ngf)
	elseif opt.which_model_netG == "resnet_512" then
		netG = defineG_resnet_512(input_nc, output_nc, ngf)
	elseif opt.which_model_netG == "uresnet_512" then 
		netG = defineG_Uresnet_512(input_nc, output_nc, ngf)
	else
		error("unsupported netG model")
	end
	netG:apply(weights_init)
	return netG
end

-- generator with encoder and decoder
function defineG_encoder_decoder(input_nc, output_nc, ngf)
	local netG = nil 
	-- input is (nc) x 256 x 256
	local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 1 x 1
	
	local d1 = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 2 x 2
	local d2 = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 4 x 4
	local d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 8 x 8
	local d4 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	local d5 = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	local d6 = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf) x128 x 128
	local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)
	-- input is (nc) x 256 x 256
	
	local o1 = d8 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})

	return netG
end

-- generator with encoder, decoder and skip connections
function defineG_unet(input_nc, output_nc, ngf)
	local netG = nil
	-- input is (nc) x 256 x 256
	local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 1 x 1
	
	local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 2 x 2
	local d1 = {d1_,e7} - nn.JoinTable(2)
	local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 4 x 4
	local d2 = {d2_,e6} - nn.JoinTable(2)
	local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 8 x 8
	local d3 = {d3_,e5} - nn.JoinTable(2)
	local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	local d4 = {d4_,e4} - nn.JoinTable(2)
	local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	local d5 = {d5_,e3} - nn.JoinTable(2)
	local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	local d6 = {d6_,e2} - nn.JoinTable(2)
	local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf) x128 x 128
	local d7 = {d7_,e1} - nn.JoinTable(2)
	local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
	-- input is (nc) x 256 x 256
	
	local o1 = d8 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})
	
	--graph.dot(netG.fg,'netG','unet')
	--graph.dot(netG.fg,'netG')

	return netG
end

-- generator with encoder, decoder and skip connections
-- decoder has upsampling + stride 1 convolution rather than convolutions with stride 1/2
function defineG_unet_upsampling(input_nc, output_nc, ngf)
	local netG = nil
	-- input is (nc) x 256 x 256
	local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 1 x 1
	
	local d1_ = e8 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 1, 1, 1, 1) - nn.SpatialZeroPadding(0, 1, 0, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 2 x 2
	local d1 = {d1_,e7} - nn.JoinTable(2)
	-- input is (ngf * 8 * 2) x 2 x 2
	local d2_ = d1 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -  nn.SpatialConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 1, 1, 1, 1) - nn.SpatialZeroPadding(1, 0, 1, 0) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 4 x 4
	local d2 = {d2_,e6} - nn.JoinTable(2)
	-- input is (ngf * 8 * 2) x 4 x 4
	local d3_ = d2 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - nn.SpatialConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 1, 1, 1, 1) - nn.SpatialZeroPadding(1, 0, 0, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 8 x 8
	local d3 = {d3_,e5} - nn.JoinTable(2)
	-- input is (ngf * 8 * 2) x 8 x 8
	local d4_ = d3 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - nn.SpatialConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 1, 1, 1, 1) - nn.SpatialZeroPadding(0, 1, 1, 0) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	local d4 = {d4_,e4} - nn.JoinTable(2)
	-- input is (ngf * 8 * 2) x 16 x 16
	local d5_ = d4 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - nn.SpatialConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 1, 1, 1, 1) - nn.SpatialZeroPadding(0, 1, 0, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	local d5 = {d5_,e3} - nn.JoinTable(2)
	-- input is (ngf * 4 * 2) x 32 x 32
	local d6_ = d5 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - nn.SpatialConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 1, 1, 1, 1) - nn.SpatialZeroPadding(1, 0, 1, 0) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	local d6 = {d6_,e2} - nn.JoinTable(2)
	-- input is (ngf * 2 * 2) x 64 x 64
	local d7_ = d6 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - nn.SpatialConvolution(ngf * 2 * 2, ngf, 4, 4, 1, 1, 1, 1) - nn.SpatialZeroPadding(1, 0, 0, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf) x128 x 128
	local d7 = {d7_,e1} - nn.JoinTable(2)
	-- input is (ngf * 2) x128 x 128
	local d8 = d7 - nn.ReLU(true) - nn.SpatialUpSamplingNearest(2) - nn.SpatialConvolution(ngf * 2, output_nc, 4, 4, 1, 1, 1, 1) - nn.SpatialZeroPadding(0, 1, 1, 0)
	-- input is (nc) x 256 x 256
	
	local o1 = d8 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})
	
	--graph.dot(netG.fg,'netG','unet_up') --bbescos
	--graph.dot(netG.fg,'netG')

	return netG
end

-- generator with encoder, decoder and skip connections
function defineG_unet_128(input_nc, output_nc, ngf)
	-- Two layer less than the default unet to handle 128x128 input
	local netG = nil
	-- input is (nc) x 128 x 128
	local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 64 x 64
	local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 32 x 32
	local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 16 x 16
	local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 1 x 1
	
	local d1_ = e7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 2 x 2
	local d1 = {d1_,e6} - nn.JoinTable(2)
	local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 4 x 4
	local d2 = {d2_,e5} - nn.JoinTable(2)
	local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 8 x 8
	local d3 = {d3_,e4} - nn.JoinTable(2)
	local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 8) x 16 x 16
	local d4 = {d4_,e3} - nn.JoinTable(2)
	local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 4) x 32 x 32
	local d5 = {d5_,e2} - nn.JoinTable(2)
	local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf * 2) x 64 x 64
	local d6 = {d6_,e1} - nn.JoinTable(2)
	local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x128 x 128
	
	local o1 = d7 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})
	
	--graph.dot(netG.fg,'netG')
	
	return netG
end

-- definition of one ResNet block
local function resnetBlock(dim, padding_type)
	convBlock = nn.Sequential()
	local padding = 0
	if padding_type == 'reflect' then
		convBlock:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
	elseif padding_type == 'replicate' then
		convBlock:add(nn.SpatialReplicatePadding(1, 1, 1, 1))
	elseif padding_type == 'zero' then
		padding = 1
	end

	convBlock:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, padding, padding))
	convBlock:add(normalization(dim))
	convBlock:add(nn.ReLU(true))
	if padding_type == 'reflect' then
		convBlock:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
	elseif padding_type == 'replicate' then
		convBlock:add(nn.SpatialReplicatePadding(1, 1, 1, 1))
	end
	convBlock:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, padding, padding))
	convBlock:add(normalization(dim))

	local concat = nn.ConcatTable()
	concat:add(convBlock)
	concat:add(nn.Identity())

	local resBlock = nn.Sequential()
	resBlock:add(concat):add(nn.CAddTable())

	return resBlock
end

-- generator with encoder, 6 ResNet blocks and decoder
function defineG_resnet_512(input_nc, output_nc, ngf)
	local netG = nil
	padding_type = 'reflect'

	-- input is (nc) x 512 x 512
	local e1_ = - nn.Identity()
	local e1 = e1_ - nn.SpatialReflectionPadding(3, 3, 3, 3) - nn.SpatialConvolution(input_nc, ngf, 7, 7, 1, 1) - normalization(ngf)
	-- input is (nc) x 512 x 512
	local e2 = e1 - nn.ReLU(true) - nn.SpatialConvolution(ngf, ngf*2, 3, 3, 2, 2, 1, 1) - normalization(ngf*2)
	-- input is (nc) x 256 x 256
	local e3 = e2 - nn.ReLU(true) - nn.SpatialConvolution(ngf*2, ngf*4, 3, 3, 2, 2, 1, 1) - normalization(ngf*4)
	-- input is (nc) x 128 x 128
	local e4 = e3 - nn.ReLU(true) - nn.SpatialConvolution(ngf*4, ngf*8, 3, 3, 2, 2, 1, 1) - normalization(ngf*8)
	-- input is (nc) x 64 x 64
	local d1 = e4 - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type)
	-- input is (nc) x 64 x 64
	local d2 = d1 - nn.SpatialFullConvolution(ngf*8, ngf*4, 3, 3, 2, 2, 1, 1, 1, 1) - normalization(ngf*4)
	-- input is (nc) x 128 x 128
	local d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf*4, ngf*2, 3, 3, 2, 2, 1, 1, 1, 1) - normalization(ngf*2)
	-- input is (nc) x 256 x 256
	local d4 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf*2, ngf, 3, 3, 2, 2, 1, 1, 1, 1) - normalization(ngf)
	-- input is (nc) x 512 x 512
	local d5 = d4 - nn.ReLU(true) - nn.SpatialReflectionPadding(3, 3, 3, 3) - nn.SpatialConvolution(ngf, output_nc, 7, 7, 1, 1) - nn.Tanh()

	netG = nn.gModule({e1_},{d5})

	return netG
end

-- generator with encoder, 6 ResNet blocks, decoder and skip connections
function defineG_Uresnet_512(input_nc, output_nc, ngf)
	local netG = nil
	padding_type = 'reflect'

	-- input is (nc) x 512 x 512
	local e1_ = - nn.Identity()
	local e1 = e1_ - nn.SpatialReflectionPadding(3, 3, 3, 3) - nn.SpatialConvolution(input_nc, ngf, 7, 7, 1, 1) - normalization(ngf)
	-- input is (nc) x 512 x 512
	local e2 = e1 - nn.ReLU(true) - nn.SpatialConvolution(ngf, ngf*2, 3, 3, 2, 2, 1, 1) - normalization(ngf*2)
	-- input is (nc) x 256 x 256
	local e3 = e2 - nn.ReLU(true) - nn.SpatialConvolution(ngf*2, ngf*4, 3, 3, 2, 2, 1, 1) - normalization(ngf*4)
	-- input is (nc) x 128 x 128
	local e4 = e3 - nn.ReLU(true) - nn.SpatialConvolution(ngf*4, ngf*8, 3, 3, 2, 2, 1, 1) - normalization(ngf*8)
	-- input is (nc) x 64 x 64
	local d1 = e4 - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type)
	-- input is (nc) x 64 x 64
	local d2_ = d1 - nn.SpatialFullConvolution(ngf*8, ngf*4, 3, 3, 2, 2, 1, 1, 1, 1) - normalization(ngf*4)
	local d2 = {d2_,e3} - nn.JoinTable(2)
	-- input is (nc) x 128 x 128
	local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf*2, 3, 3, 2, 2, 1, 1, 1, 1) - normalization(ngf*2)
	local d3 = {d3_,e2} - nn.JoinTable(2)
	-- input is (nc) x 256 x 256
	local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 3, 3, 2, 2, 1, 1, 1, 1) - normalization(ngf)
	local d4 = {d4_,e1} - nn.JoinTable(2)
	-- input is (nc) x 512 x 512
	local d5 = d4 - nn.ReLU(true) - nn.SpatialReflectionPadding(3, 3, 3, 3) - nn.SpatialConvolution(ngf * 2, output_nc, 7, 7, 1, 1) - nn.Tanh()

	netG = nn.gModule({e1_},{d5})

	--graph.dot(netG.fg,'netG','uResNet')
	--graph.dot(netG.fg,'netG')

	return netG
end

-- generator with encoder, 6 ResNet blocks and decoder
function defineG_resnet_256(input_nc, output_nc, ngf)
	local netG = nil
	padding_type = 'reflect'

	-- input is (nc) x 256 x 256
	local e1_ = - nn.Identity()
	local e1 = e1_ - nn.SpatialReflectionPadding(3, 3, 3, 3) - nn.SpatialConvolution(input_nc, ngf, 7, 7, 1, 1) - normalization(ngf)
	-- input is (nc) x 256 x 256
	local e2 = e1 - nn.ReLU(true) - nn.SpatialConvolution(ngf, ngf*2, 3, 3, 2, 2, 1, 1) - normalization(ngf*2)
	-- input is (nc) x 128 x 128
	local e3 = e2 - nn.ReLU(true) - nn.SpatialConvolution(ngf*2, ngf*4, 3, 3, 2, 2, 1, 1) - normalization(ngf*4)
	-- input is (nc) x 64 x 64
	local e4 = e3 - nn.ReLU(true) - nn.SpatialConvolution(ngf*4, ngf*8, 3, 3, 2, 2, 1, 1) - normalization(ngf*8)
	-- input is (nc) x 32 x 32
	local d1 = e4 - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type) - resnetBlock(ngf*8, padding_type)
	-- input is (nc) x 32 x 32
	local d2 = d1 - nn.SpatialFullConvolution(ngf*8, ngf*4, 3, 3, 2, 2, 1, 1, 1, 1) - normalization(ngf*4)
	-- input is (nc) x 64 x 64
	local d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf*4, ngf*2, 3, 3, 2, 2, 1, 1, 1, 1) - normalization(ngf*2)
	-- input is (nc) x 128 x 128
	local d4 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf*2, ngf, 3, 3, 2, 2, 1, 1, 1, 1) - normalization(ngf)
	-- input is (nc) x 256 x 256
	local d5 = d4 - nn.SpatialReflectionPadding(3, 3, 3, 3) - nn.SpatialConvolution(ngf, output_nc, 7, 7, 1, 1) - nn.Tanh()

	netG = nn.gModule({e1_},{d5})

	return netG
end

-- discriminator definition
function defineD_basic(input_nc, output_nc, ndf)
	n_layers = 3
	return defineD_n_layers(input_nc, output_nc, ndf, n_layers)
end

-- discriminator at pixel level
function defineD_pixelGAN(input_nc, output_nc, ndf)
	local netD = nn.Sequential()
	
	-- input is (nc) x 256 x 256
	netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 1, 1, 1, 1, 0, 0))
	netD:add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf) x 256 x 256
	netD:add(nn.SpatialConvolution(ndf, ndf * 2, 1, 1, 1, 1, 0, 0))
	netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*2) x 256 x 256
	netD:add(nn.SpatialConvolution(ndf * 2, 1, 1, 1, 1, 1, 0, 0))
	-- state size: 1 x 256 x 256
	netD:add(nn.Sigmoid())
	-- state size: 1 x 256 x 256

	return netD
end

-- if n=0, then use pixelGAN (rf=1)
-- else rf is 16 if n=1
--            34 if n=2
--            70 if n=3
--            142 if n=4
--            286 if n=5
--            574 if n=6
function defineD_n_layers(input_nc, output_nc, ndf, n_layers)
	if n_layers==0 then
		return defineD_pixelGAN(input_nc, output_nc, ndf)
	else
	
		local netD = nn.Sequential()
		
		-- input is (nc) x 256 x 256
		netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
		netD:add(nn.LeakyReLU(0.2, true))
		-- input is (nc) x 128 x 128
		local nf_mult = 1
		local nf_mult_prev = 1
		for n = 1, n_layers - 1 do 
			nf_mult_prev = nf_mult
			nf_mult = math.min(2^n,8)
			netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
			netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
		end
		-- input is (nc) x 32 x 32
		-- state size: (ndf*M) x N x N
		nf_mult_prev = nf_mult
		nf_mult = math.min(2^n_layers,8)
		netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 1, 1, 1, 1))
		netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
		-- state size: (ndf*M*2) x (N-1) x (N-1)
		netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 4, 4, 1, 1, 1, 1))
		-- state size: 1 x (N-2) x (N-2)
		
		netD:add(nn.Sigmoid())
		-- state size: 1 x (N-2) x (N-2)
		
		return netD
	end
end

local pattern = torch.Tensor({{8,-3, 9,5},--mean (0), correlation (0){
	{4,2, 7,-12},--mean (1.12461e-05), correlation (0.0437584){
	{-11,9, -8,2},--mean (3.37382e-05), correlation (0.0617409){
	{7,-12, 12,-13},--mean (5.62303e-05), correlation (0.0636977){
	{2,-13, 2,12},--mean (0.000134953), correlation (0.085099){
	{1,-7, 1,6},--mean (0.000528565), correlation (0.0857175){
	{-2,-10, -2,-4},--mean (0.0188821), correlation (0.0985774){
	{-13,-13, -11,-8},--mean (0.0363135), correlation (0.0899616){
	{-13,-3, -12,-9},--mean (0.121806), correlation (0.099849){
	{10,4, 11,9},--mean (0.122065), correlation (0.093285){
	{-13,-8, -8,-9},--mean (0.162787), correlation (0.0942748){
	{-11,7, -9,12},--mean (0.21561), correlation (0.0974438){
	{7,7, 12,6},--mean (0.160583), correlation (0.130064){
	{-4,-5, -3,0},--mean (0.228171), correlation (0.132998){
	{-13,2, -12,-3},--mean (0.00997526), correlation (0.145926){
	{-9,0, -7,5},--mean (0.198234), correlation (0.143636){
	{12,-6, 12,-1},--mean (0.0676226), correlation (0.16689){
	{-3,6, -2,12},--mean (0.166847), correlation (0.171682){
	{-6,-13, -4,-8},--mean (0.101215), correlation (0.179716){
	{11,-13, 12,-8},--mean (0.200641), correlation (0.192279){
	{4,7, 5,1},--mean (0.205106), correlation (0.186848){
	{5,-3, 10,-3},--mean (0.234908), correlation (0.192319){
	{3,-7, 6,12},--mean (0.0709964), correlation (0.210872){
	{-8,-7, -6,-2},--mean (0.0939834), correlation (0.212589){
	{-2,11, -1,-10},--mean (0.127778), correlation (0.20866){
	{-13,12, -8,10},--mean (0.14783), correlation (0.206356){
	{-7,3, -5,-3},--mean (0.182141), correlation (0.198942){
	{-4,2, -3,7},--mean (0.188237), correlation (0.21384){
	{-10,-12, -6,11},--mean (0.14865), correlation (0.23571){
	{5,-12, 6,-7},--mean (0.222312), correlation (0.23324){
	{5,-6, 7,-1},--mean (0.229082), correlation (0.23389){
	{1,0, 4,-5},--mean (0.241577), correlation (0.215286){
	{9,11, 11,-13},--mean (0.00338507), correlation (0.251373){
	{4,7, 4,12},--mean (0.131005), correlation (0.257622){
	{2,-1, 4,4},--mean (0.152755), correlation (0.255205){
	{-4,-12, -2,7},--mean (0.182771), correlation (0.244867){
	{-8,-5, -7,-10},--mean (0.186898), correlation (0.23901){
	{4,11, 9,12},--mean (0.226226), correlation (0.258255){
	{0,-8, 1,-13},--mean (0.0897886), correlation (0.274827){
	{-13,-2, -8,2},--mean (0.148774), correlation (0.28065){
	{-3,-2, -2,3},--mean (0.153048), correlation (0.283063){
	{-6,9, -4,-9},--mean (0.169523), correlation (0.278248){
	{8,12, 10,7},--mean (0.225337), correlation (0.282851){
	{0,9, 1,3},--mean (0.226687), correlation (0.278734){
	{7,-5, 11,-10},--mean (0.00693882), correlation (0.305161){
	{-13,-6, -11,0},--mean (0.0227283), correlation (0.300181){
	{10,7, 12,1},--mean (0.125517), correlation (0.31089){
	{-6,-3, -6,12},--mean (0.131748), correlation (0.312779){
	{10,-9, 12,-4},--mean (0.144827), correlation (0.292797){
	{-13,8, -8,-12},--mean (0.149202), correlation (0.308918){
	{-13,0, -8,-4},--mean (0.160909), correlation (0.310013){
	{3,3, 7,8},--mean (0.177755), correlation (0.309394){
	{5,7, 10,-7},--mean (0.212337), correlation (0.310315){
	{-1,7, 1,-12},--mean (0.214429), correlation (0.311933){
	{3,-10, 5,6},--mean (0.235807), correlation (0.313104){
	{2,-4, 3,-10},--mean (0.00494827), correlation (0.344948){
	{-13,0, -13,5},--mean (0.0549145), correlation (0.344675){
	{-13,-7, -12,12},--mean (0.103385), correlation (0.342715){
	{-13,3, -11,8},--mean (0.134222), correlation (0.322922){
	{-7,12, -4,7},--mean (0.153284), correlation (0.337061){
	{6,-10, 12,8},--mean (0.154881), correlation (0.329257){
	{-9,-1, -7,-6},--mean (0.200967), correlation (0.33312){
	{-2,-5, 0,12},--mean (0.201518), correlation (0.340635){
	{-12,5, -7,5},--mean (0.207805), correlation (0.335631){
	{3,-10, 8,-13},--mean (0.224438), correlation (0.34504){
	{-7,-7, -4,5},--mean (0.239361), correlation (0.338053){
	{-3,-2, -1,-7},--mean (0.240744), correlation (0.344322){
	{2,9, 5,-11},--mean (0.242949), correlation (0.34145){
	{-11,-13, -5,-13},--mean (0.244028), correlation (0.336861){
	{-1,6, 0,-1},--mean (0.247571), correlation (0.343684){
	{5,-3, 5,2},--mean (0.000697256), correlation (0.357265){
	{-4,-13, -4,12},--mean (0.00213675), correlation (0.373827){
	{-9,-6, -9,6},--mean (0.0126856), correlation (0.373938){
	{-12,-10, -8,-4},--mean (0.0152497), correlation (0.364237){
	{10,2, 12,-3},--mean (0.0299933), correlation (0.345292){
	{7,12, 12,12},--mean (0.0307242), correlation (0.366299){
	{-7,-13, -6,5},--mean (0.0534975), correlation (0.368357){
	{-4,9, -3,4},--mean (0.099865), correlation (0.372276){
	{7,-1, 12,2},--mean (0.117083), correlation (0.364529){
	{-7,6, -5,1},--mean (0.126125), correlation (0.369606){
	{-13,11, -12,5},--mean (0.130364), correlation (0.358502){
	{-3,7, -2,-6},--mean (0.131691), correlation (0.375531){
	{7,-8, 12,-7},--mean (0.160166), correlation (0.379508){
	{-13,-7, -11,-12},--mean (0.167848), correlation (0.353343){
	{1,-3, 12,12},--mean (0.183378), correlation (0.371916){
	{2,-6, 3,0},--mean (0.228711), correlation (0.371761){
	{-4,3, -2,-13},--mean (0.247211), correlation (0.364063){
	{-1,-13, 1,9},--mean (0.249325), correlation (0.378139){
	{7,1, 8,-6},--mean (0.000652272), correlation (0.411682){
	{1,-1, 3,12},--mean (0.00248538), correlation (0.392988){
	{9,1, 12,6},--mean (0.0206815), correlation (0.386106){
	{-1,-9, -1,3},--mean (0.0364485), correlation (0.410752){
	{-13,-13, -10,5},--mean (0.0376068), correlation (0.398374){
	{7,7, 10,12},--mean (0.0424202), correlation (0.405663){
	{12,-5, 12,9},--mean (0.0942645), correlation (0.410422){
	{6,3, 7,11},--mean (0.1074), correlation (0.413224){
	{5,-13, 6,10},--mean (0.109256), correlation (0.408646){
	{2,-12, 2,3},--mean (0.131691), correlation (0.416076){
	{3,8, 4,-6},--mean (0.165081), correlation (0.417569){
	{2,6, 12,-13},--mean (0.171874), correlation (0.408471){
	{9,-12, 10,3},--mean (0.175146), correlation (0.41296){
	{-8,4, -7,9},--mean (0.183682), correlation (0.402956){
	{-11,12, -4,-6},--mean (0.184672), correlation (0.416125){
	{1,12, 2,-8},--mean (0.191487), correlation (0.386696){
	{6,-9, 7,-4},--mean (0.192668), correlation (0.394771){
	{2,3, 3,-2},--mean (0.200157), correlation (0.408303){
	{6,3, 11,0},--mean (0.204588), correlation (0.411762){
	{3,-3, 8,-8},--mean (0.205904), correlation (0.416294){
	{7,8, 9,3},--mean (0.213237), correlation (0.409306){
	{-11,-5, -6,-4},--mean (0.243444), correlation (0.395069){
	{-10,11, -5,10},--mean (0.247672), correlation (0.413392){
	{-5,-8, -3,12},--mean (0.24774), correlation (0.411416){
	{-10,5, -9,0},--mean (0.00213675), correlation (0.454003){
	{8,-1, 12,-6},--mean (0.0293635), correlation (0.455368){
	{4,-6, 6,-11},--mean (0.0404971), correlation (0.457393){
	{-10,12, -8,7},--mean (0.0481107), correlation (0.448364){
	{4,-2, 6,7},--mean (0.050641), correlation (0.455019){
	{-2,0, -2,12},--mean (0.0525978), correlation (0.44338){
	{-5,-8, -5,2},--mean (0.0629667), correlation (0.457096){
	{7,-6, 10,12},--mean (0.0653846), correlation (0.445623){
	{-9,-13, -8,-8},--mean (0.0858749), correlation (0.449789){
	{-5,-13, -5,-2},--mean (0.122402), correlation (0.450201){
	{8,-8, 9,-13},--mean (0.125416), correlation (0.453224){
	{-9,-11, -9,0},--mean (0.130128), correlation (0.458724){
	{1,-8, 1,-2},--mean (0.132467), correlation (0.440133){
	{7,-4, 9,1},--mean (0.132692), correlation (0.454){
	{-2,1, -1,-4},--mean (0.135695), correlation (0.455739){
	{11,-6, 12,-11},--mean (0.142904), correlation (0.446114){
	{-12,-9, -6,4},--mean (0.146165), correlation (0.451473){
	{3,7, 7,12},--mean (0.147627), correlation (0.456643){
	{5,5, 10,8},--mean (0.152901), correlation (0.455036){
	{0,-4, 2,8},--mean (0.167083), correlation (0.459315){
	{-9,12, -5,-13},--mean (0.173234), correlation (0.454706){
	{0,7, 2,12},--mean (0.18312), correlation (0.433855){
	{-1,2, 1,7},--mean (0.185504), correlation (0.443838){
	{5,11, 7,-9},--mean (0.185706), correlation (0.451123){
	{3,5, 6,-8},--mean (0.188968), correlation (0.455808){
	{-13,-4, -8,9},--mean (0.191667), correlation (0.459128){
	{-5,9, -3,-3},--mean (0.193196), correlation (0.458364){
	{-4,-7, -3,-12},--mean (0.196536), correlation (0.455782){
	{6,5, 8,0},--mean (0.1972), correlation (0.450481){
	{-7,6, -6,12},--mean (0.199438), correlation (0.458156){
	{-13,6, -5,-2},--mean (0.211224), correlation (0.449548){
	{1,-10, 3,10},--mean (0.211718), correlation (0.440606){
	{4,1, 8,-4},--mean (0.213034), correlation (0.443177){
	{-2,-2, 2,-13},--mean (0.234334), correlation (0.455304){
	{2,-12, 12,12},--mean (0.235684), correlation (0.443436){
	{-2,-13, 0,-6},--mean (0.237674), correlation (0.452525){
	{4,1, 9,3},--mean (0.23962), correlation (0.444824){
	{-6,-10, -3,-5},--mean (0.248459), correlation (0.439621){
	{-3,-13, -1,1},--mean (0.249505), correlation (0.456666){
	{7,5, 12,-11},--mean (0.00119208), correlation (0.495466){
	{4,-2, 5,-7},--mean (0.00372245), correlation (0.484214){
	{-13,9, -9,-5},--mean (0.00741116), correlation (0.499854){
	{7,1, 8,6},--mean (0.0208952), correlation (0.499773){
	{7,-8, 7,6},--mean (0.0220085), correlation (0.501609){
	{-7,-4, -7,1},--mean (0.0233806), correlation (0.496568){
	{-8,11, -7,-8},--mean (0.0236505), correlation (0.489719){
	{-13,6, -12,-8},--mean (0.0268781), correlation (0.503487){
	{2,4, 3,9},--mean (0.0323324), correlation (0.501938){
	{10,-5, 12,3},--mean (0.0399235), correlation (0.494029){
	{-6,-5, -6,7},--mean (0.0420153), correlation (0.486579){
	{8,-3, 9,-8},--mean (0.0548021), correlation (0.484237){
	{2,-12, 2,8},--mean (0.0616622), correlation (0.496642){
	{-11,-2, -10,3},--mean (0.0627755), correlation (0.498563){
	{-12,-13, -7,-9},--mean (0.0829622), correlation (0.495491){
	{-11,0, -10,-5},--mean (0.0843342), correlation (0.487146){
	{5,-3, 11,8},--mean (0.0929937), correlation (0.502315){
	{-2,-13, -1,12},--mean (0.113327), correlation (0.48941){
	{-1,-8, 0,9},--mean (0.132119), correlation (0.467268){
	{-13,-11, -12,-5},--mean (0.136269), correlation (0.498771){
	{-10,-2, -10,11},--mean (0.142173), correlation (0.498714){
	{-3,9, -2,-13},--mean (0.144141), correlation (0.491973){
	{2,-3, 3,2},--mean (0.14892), correlation (0.500782){
	{-9,-13, -4,0},--mean (0.150371), correlation (0.498211){
	{-4,6, -3,-10},--mean (0.152159), correlation (0.495547){
	{-4,12, -2,-7},--mean (0.156152), correlation (0.496925){
	{-6,-11, -4,9},--mean (0.15749), correlation (0.499222){
	{6,-3, 6,11},--mean (0.159211), correlation (0.503821){
	{-13,11, -5,5},--mean (0.162427), correlation (0.501907){
	{11,11, 12,6},--mean (0.16652), correlation (0.497632){
	{7,-5, 12,-2},--mean (0.169141), correlation (0.484474){
	{-1,12, 0,7},--mean (0.169456), correlation (0.495339){
	{-4,-8, -3,-2},--mean (0.171457), correlation (0.487251){
	{-7,1, -6,7},--mean (0.175), correlation (0.500024){
	{-13,-12, -8,-13},--mean (0.175866), correlation (0.497523){
	{-7,-2, -6,-8},--mean (0.178273), correlation (0.501854){
	{-8,5, -6,-9},--mean (0.181107), correlation (0.494888){
	{-5,-1, -4,5},--mean (0.190227), correlation (0.482557){
	{-13,7, -8,10},--mean (0.196739), correlation (0.496503){
	{1,5, 5,-13},--mean (0.19973), correlation (0.499759){
	{1,0, 10,-13},--mean (0.204465), correlation (0.49873){
	{9,12, 10,-1},--mean (0.209334), correlation (0.49063){
	{5,-8, 10,-9},--mean (0.211134), correlation (0.503011){
	{-1,11, 1,-13},--mean (0.212), correlation (0.499414){
	{-9,-3, -6,2},--mean (0.212168), correlation (0.480739){
	{-1,-10, 1,12},--mean (0.212731), correlation (0.502523){
	{-13,1, -8,-10},--mean (0.21327), correlation (0.489786){
	{8,-11, 10,-6},--mean (0.214159), correlation (0.488246){
	{2,-13, 3,-6},--mean (0.216993), correlation (0.50287){
	{7,-13, 12,-9},--mean (0.223639), correlation (0.470502){
	{-10,-10, -5,-7},--mean (0.224089), correlation (0.500852){
	{-10,-8, -8,-13},--mean (0.228666), correlation (0.502629){
	{4,-6, 8,5},--mean (0.22906), correlation (0.498305){
	{3,12, 8,-13},--mean (0.233378), correlation (0.503825){
	{-4,2, -3,-3},--mean (0.234323), correlation (0.476692){
	{5,-13, 10,-12},--mean (0.236392), correlation (0.475462){
	{4,-13, 5,-1},--mean (0.236842), correlation (0.504132){
	{-9,9, -4,3},--mean (0.236977), correlation (0.497739){
	{0,3, 3,-9},--mean (0.24314), correlation (0.499398){
	{-12,1, -6,1},--mean (0.243297), correlation (0.489447){
	{3,2, 4,-8},--mean (0.00155196), correlation (0.553496){
	{-10,-10, -10,9},--mean (0.00239541), correlation (0.54297){
	{8,-13, 12,12},--mean (0.0034413), correlation (0.544361){
	{-8,-12, -6,-5},--mean (0.003565), correlation (0.551225){
	{2,2, 3,7},--mean (0.00835583), correlation (0.55285){
	{10,6, 11,-8},--mean (0.00885065), correlation (0.540913){
	{6,8, 8,-12},--mean (0.0101552), correlation (0.551085){
	{-7,10, -6,5},--mean (0.0102227), correlation (0.533635){
	{-3,-9, -3,9},--mean (0.0110211), correlation (0.543121){
	{-1,-13, -1,5},--mean (0.0113473), correlation (0.550173){
	{-3,-7, -3,4},--mean (0.0140913), correlation (0.554774){
	{-8,-2, -8,3},--mean (0.017049), correlation (0.55461){
	{4,2, 12,12},--mean (0.01778), correlation (0.546921){
	{2,-5, 3,11},--mean (0.0224022), correlation (0.549667){
	{6,-9, 11,-13},--mean (0.029161), correlation (0.546295){
	{3,-1, 7,12},--mean (0.0303081), correlation (0.548599){
	{11,-1, 12,4},--mean (0.0355151), correlation (0.523943){
	{-3,0, -3,6},--mean (0.0417904), correlation (0.543395){
	{4,-11, 4,12},--mean (0.0487292), correlation (0.542818){
	{2,-4, 2,1},--mean (0.0575124), correlation (0.554888){
	{-10,-6, -8,1},--mean (0.0594242), correlation (0.544026){
	{-13,7, -11,1},--mean (0.0597391), correlation (0.550524){
	{-13,12, -11,-13},--mean (0.0608974), correlation (0.55383){
	{6,0, 11,-13},--mean (0.065126), correlation (0.552006){
	{0,-1, 1,4},--mean (0.074224), correlation (0.546372){
	{-13,3, -9,-2},--mean (0.0808592), correlation (0.554875){
	{-9,8, -6,-3},--mean (0.0883378), correlation (0.551178){
	{-13,-6, -8,-2},--mean (0.0901035), correlation (0.548446){
	{5,-9, 8,10},--mean (0.0949843), correlation (0.554694){
	{2,7, 3,-9},--mean (0.0994152), correlation (0.550979){
	{-1,-6, -1,-1},--mean (0.10045), correlation (0.552714){
	{9,5, 11,-2},--mean (0.100686), correlation (0.552594){
	{11,-3, 12,-8},--mean (0.101091), correlation (0.532394){
	{3,0, 3,5},--mean (0.101147), correlation (0.525576){
	{-1,4, 0,10},--mean (0.105263), correlation (0.531498){
	{3,-6, 4,5},--mean (0.110785), correlation (0.540491){
	{-13,0, -10,5},--mean (0.112798), correlation (0.536582){
	{5,8, 12,11},--mean (0.114181), correlation (0.555793){
	{8,9, 9,-6},--mean (0.117431), correlation (0.553763){
	{7,-4, 8,-12},--mean (0.118522), correlation (0.553452){
	{-10,4, -10,9},--mean (0.12094), correlation (0.554785){
	{7,3, 12,4},--mean (0.122582), correlation (0.555825){
	{9,-7, 10,-2},--mean (0.124978), correlation (0.549846){
	{7,0, 12,-2},--mean (0.127002), correlation (0.537452){
	{-1,-6, 0,-11}})--mean (0.127148), correlation (0.547401)

function FASTKernels()

    local kernel_stack = torch.Tensor(16, 1, 7, 7)

    local kernel1 = torch.Tensor(7,7):zero()
    kernel1[4][4] = 1
    kernel1[1][3] = -1/12
    kernel1[1][4] = -1/12
    kernel1[1][5] = -1/12
    kernel1[2][6] = -1/12
    kernel1[3][7] = -1/12
    kernel1[4][7] = -1/12
    kernel1[5][7] = -1/12
    kernel1[6][6] = -1/12
    kernel1[7][5] = -1/12
    kernel1[7][4] = -1/12
    kernel1[7][3] = -1/12
    kernel1[6][2] = -1/12

    kernel_stack[1][1] = kernel1

    local kernel2 = torch.Tensor(7,7):zero()
    kernel2[4][4] = 1
    kernel2[1][4] = -1/12
    kernel2[1][5] = -1/12
    kernel2[2][6] = -1/12
    kernel2[3][7] = -1/12
    kernel2[4][7] = -1/12
    kernel2[5][7] = -1/12
    kernel2[6][6] = -1/12
    kernel2[7][5] = -1/12
    kernel2[7][4] = -1/12
    kernel2[7][3] = -1/12
    kernel2[6][2] = -1/12
    kernel2[5][1] = -1/12

    kernel_stack[2][1] = kernel2

    local kernel3 = torch.Tensor(7,7):zero()
    kernel3[4][4] = 1
    kernel3[1][5] = -1/12
    kernel3[2][6] = -1/12
    kernel3[3][7] = -1/12
    kernel3[4][7] = -1/12
    kernel3[5][7] = -1/12
    kernel3[6][6] = -1/12
    kernel3[7][5] = -1/12
    kernel3[7][4] = -1/12
    kernel3[7][3] = -1/12
    kernel3[6][2] = -1/12
    kernel3[5][1] = -1/12
    kernel3[4][1] = -1/12

    kernel_stack[3][1] = kernel3

    local kernel4 = torch.Tensor(7,7):zero()
    kernel4[4][4] = 1
    kernel4[2][6] = -1/12
    kernel4[3][7] = -1/12
    kernel4[4][7] = -1/12
    kernel4[5][7] = -1/12
    kernel4[6][6] = -1/12
    kernel4[7][5] = -1/12
    kernel4[7][4] = -1/12
    kernel4[7][3] = -1/12
    kernel4[6][2] = -1/12
    kernel4[5][1] = -1/12
    kernel4[4][1] = -1/12
    kernel4[3][1] = -1/12

    kernel_stack[4][1] = kernel4

    local kernel5 = torch.Tensor(7,7):zero()
    kernel5[4][4] = 1
    kernel5[3][7] = -1/12
    kernel5[4][7] = -1/12
    kernel5[5][7] = -1/12
    kernel5[6][6] = -1/12
    kernel5[7][5] = -1/12
    kernel5[7][4] = -1/12
    kernel5[7][3] = -1/12
    kernel5[6][2] = -1/12
    kernel5[5][1] = -1/12
    kernel5[4][1] = -1/12
    kernel5[3][1] = -1/12
    kernel5[2][2] = -1/12

    kernel_stack[5][1] = kernel5

    local kernel6 = torch.Tensor(7,7):zero()
    kernel6[4][4] = 1
    kernel6[4][7] = -1/12
    kernel6[5][7] = -1/12
    kernel6[6][6] = -1/12
    kernel6[7][5] = -1/12
    kernel6[7][4] = -1/12
    kernel6[7][3] = -1/12
    kernel6[6][2] = -1/12
    kernel6[5][1] = -1/12
    kernel6[4][1] = -1/12
    kernel6[3][1] = -1/12
    kernel6[2][2] = -1/12
    kernel6[1][3] = -1/12

    kernel_stack[6][1] = kernel6

    local kernel7 = torch.Tensor(7,7):zero()
    kernel7[4][4] = 1
    kernel7[5][7] = -1/12
    kernel7[6][6] = -1/12
    kernel7[7][5] = -1/12
    kernel7[7][4] = -1/12
    kernel7[7][3] = -1/12
    kernel7[6][2] = -1/12
    kernel7[5][1] = -1/12
    kernel7[4][1] = -1/12
    kernel7[3][1] = -1/12
    kernel7[2][2] = -1/12
    kernel7[1][3] = -1/12
    kernel7[1][4] = -1/12

    kernel_stack[7][1] = kernel7

    local kernel8 = torch.Tensor(7,7):zero()
    kernel8[4][4] = 1
    kernel8[6][6] = -1/12
    kernel8[7][5] = -1/12
    kernel8[7][4] = -1/12
    kernel8[7][3] = -1/12
    kernel8[6][2] = -1/12
    kernel8[5][1] = -1/12
    kernel8[4][1] = -1/12
    kernel8[3][1] = -1/12
    kernel8[2][2] = -1/12
    kernel8[1][3] = -1/12
    kernel8[1][4] = -1/12
    kernel8[1][5] = -1/12

    kernel_stack[8][1] = kernel8

    local kernel9 = torch.Tensor(7,7):zero()
    kernel9[4][4] = 1
    kernel9[7][5] = -1/12
    kernel9[7][4] = -1/12
    kernel9[7][3] = -1/12
    kernel9[6][2] = -1/12
    kernel9[5][1] = -1/12
    kernel9[4][1] = -1/12
    kernel9[3][1] = -1/12
    kernel9[2][2] = -1/12
    kernel9[1][3] = -1/12
    kernel9[1][4] = -1/12
    kernel9[1][5] = -1/12
    kernel9[2][6] = -1/12

    kernel_stack[9][1] = kernel9

    local kernel10 = torch.Tensor(7,7):zero()
    kernel10[4][4] = 1
    kernel10[7][4] = -1/12
    kernel10[7][3] = -1/12
    kernel10[6][2] = -1/12
    kernel10[5][1] = -1/12
    kernel10[4][1] = -1/12
    kernel10[3][1] = -1/12
    kernel10[2][2] = -1/12
    kernel10[1][3] = -1/12
    kernel10[1][4] = -1/12
    kernel10[1][5] = -1/12
    kernel10[2][6] = -1/12
    kernel10[3][7] = -1/12

    kernel_stack[10][1] = kernel10

    local kernel11 = torch.Tensor(7,7):zero()
    kernel11[4][4] = 1
    kernel11[7][3] = -1/12
    kernel11[6][2] = -1/12
    kernel11[5][1] = -1/12
    kernel11[4][1] = -1/12
    kernel11[3][1] = -1/12
    kernel11[2][2] = -1/12
    kernel11[1][3] = -1/12
    kernel11[1][4] = -1/12
    kernel11[1][5] = -1/12
    kernel11[2][6] = -1/12
    kernel11[3][7] = -1/12
    kernel11[4][7] = -1/12

    kernel_stack[11][1] = kernel11

    local kernel12 = torch.Tensor(7,7):zero()
    kernel12[4][4] = 1
    kernel12[6][2] = -1/12
    kernel12[5][1] = -1/12
    kernel12[4][1] = -1/12
    kernel12[3][1] = -1/12
    kernel12[2][2] = -1/12
    kernel12[1][3] = -1/12
    kernel12[1][4] = -1/12
    kernel12[1][5] = -1/12
    kernel12[2][6] = -1/12
    kernel12[3][7] = -1/12
    kernel12[4][7] = -1/12
    kernel12[5][7] = -1/12

    kernel_stack[12][1] = kernel12

    local kernel13 = torch.Tensor(7,7):zero()
    kernel13[4][4] = 1
    kernel13[5][1] = -1/12
    kernel13[4][1] = -1/12
    kernel13[3][1] = -1/12
    kernel13[2][2] = -1/12
    kernel13[1][3] = -1/12
    kernel13[1][4] = -1/12
    kernel13[1][5] = -1/12
    kernel13[2][6] = -1/12
    kernel13[3][7] = -1/12
    kernel13[4][7] = -1/12
    kernel13[5][7] = -1/12
    kernel13[6][6] = -1/12

    kernel_stack[13][1] = kernel13

    local kernel14 = torch.Tensor(7,7):zero()
    kernel14[4][4] = 1
    kernel14[4][1] = -1/12
    kernel14[3][1] = -1/12
    kernel14[2][2] = -1/12
    kernel14[1][3] = -1/12
    kernel14[1][4] = -1/12
    kernel14[1][5] = -1/12
    kernel14[2][6] = -1/12
    kernel14[3][7] = -1/12
    kernel14[4][7] = -1/12
    kernel14[5][7] = -1/12
    kernel14[6][6] = -1/12
    kernel14[7][5] = -1/12

    kernel_stack[14][1] = kernel14

    local kernel15 = torch.Tensor(7,7):zero()
    kernel15[4][4] = 1
    kernel15[3][1] = -1/12
    kernel15[2][2] = -1/12
    kernel15[1][3] = -1/12
    kernel15[1][4] = -1/12
    kernel15[1][5] = -1/12
    kernel15[2][6] = -1/12
    kernel15[3][7] = -1/12
    kernel15[4][7] = -1/12
    kernel15[5][7] = -1/12
    kernel15[6][6] = -1/12
    kernel15[7][5] = -1/12
    kernel15[7][4] = -1/12

    kernel_stack[15][1] = kernel15

    local kernel16 = torch.Tensor(7,7):zero()
    kernel16[4][4] = 1
    kernel16[2][2] = -1/12
    kernel16[1][3] = -1/12
    kernel16[1][4] = -1/12
    kernel16[1][5] = -1/12
    kernel16[2][6] = -1/12
    kernel16[3][7] = -1/12
    kernel16[4][7] = -1/12
    kernel16[5][7] = -1/12
    kernel16[6][6] = -1/12
    kernel16[7][5] = -1/12
    kernel16[7][4] = -1/12
    kernel16[7][3] = -1/12

    kernel_stack[16][1] = kernel16

    return kernel_stack
end

function create_orientation_kernels(kernel_size)

	-- function to compute the kenrles for moments of a patch
	
	local val = (kernel_size - 1)/2

	local kernel_stack = torch.zeros(3, 1, kernel_size, kernel_size)
	
	local kernel10 = torch.zeros(kernel_size, kernel_size)
	local kernel00 = torch.zeros(kernel_size, kernel_size)
	for i = -val, val do
		local jmax = math.sqrt(val * val - i * i);
		for j = -val, val do
			if math.abs(j) <= math.abs(jmax) then
				kernel10[i + val + 1][j + val + 1] = i
				kernel00[i + val + 1][j + val + 1] = 1
			end
		end
	end
	local kernel01 = kernel10:clone():transpose(1,2)

	kernel_stack[1][1] = kernel10
	kernel_stack[2][1] = kernel00
	kernel_stack[3][1] = kernel01

	return kernel_stack
end

function create_pattern_kernels(pattern, kernel_size)
	-- given a pattern it creates the corresponding kernels for binary test
	
	local kernel_stack = torch.zeros(pattern:size(1), 1, kernel_size, kernel_size)
	
	for i = 1, pattern:size(1) do
		local kernel = torch.zeros(kernel_size, kernel_size)
		local offset = (kernel_size -1)/2
		kernel[pattern[i][1] + offset][pattern[i][2] + offset] = -1
		kernel[pattern[i][3] + offset][pattern[i][4] + offset] = 1
		kernel_stack[i][1] = kernel
	end

	return kernel_stack
end

function define_netFDetector(stride)

	local kernel_size = 7
	local padding_size = (kernel_size - 1)/2

	local net = nn.Sequential()

	local conv1 = nn.SpatialConvolution(1, 16, kernel_size, kernel_size, stride, stride)
	conv1.weight = FASTKernels()
	conv1.bias:zero()
	local conv2 = nn.Max(2)
	local conv3 = nn.Power(2)
    local conv4 = nn.Add(1, true)
    conv4.bias:zero():add(-0.015)

	net:add(nn.SpatialReplicationPadding(padding_size, padding_size, padding_size, padding_size))
	net:add(conv1)
	net:add(conv2)
	net:add(nn.Unsqueeze(2))
	net:add(conv3)
	net:add(conv4)
	net:add(nn.Sigmoid())

	return net
end

function define_netFOrientation(stride)

	local kernel_size = 29
	local net = nn.Sequential()
	local conv1 = nn.SpatialConvolution(1, 3, kernel_size, kernel_size, stride, stride, 0, 0)
	local orientation_kernels = create_orientation_kernels(kernel_size)
	conv1.weight = orientation_kernels
	conv1.bias:zero()
	local padding_size = (kernel_size - 1)/2
	net:add(nn.SpatialReplicationPadding(padding_size, padding_size, padding_size, padding_size))
	net:add(conv1)

	return net
end

function define_netFDescriptor(stride)

	local kernel_size = 29
	local net = nn.Sequential()
	local conv1 = nn.SpatialConvolution(1, 256, kernel_size, kernel_size, stride, stride, 0, 0)
	local pattern_kernels = create_pattern_kernels(pattern, kernel_size)
	conv1.weight = pattern_kernels
	conv1.bias:zero()
	
	local padding_size = (kernel_size - 1)/2
	net:add(nn.SpatialReplicationPadding(padding_size, padding_size, padding_size, padding_size))
	net:add(conv1)
	net:add(nn.Sigmoid())

	return net
end

function define_netFeatures(lossDetector, lossOrientation, lossDescriptor, stride)

	local netFeatures = nn.DepthConcat(2)

	if lossDetector == 1 then
		local netFDetector = define_netFDetector(stride)
		if lossOrientation == 1 then
			local netFOrientation = define_netFOrientation(stride)
			if lossDescriptor == 1 then
				local netFDescriptor = define_netFDescriptor(stride)
				netFeatures:add(netFDetector)
				netFeatures:add(netFOrientation)
				netFeatures:add(netFDescriptor)
			else
				netFeatures:add(netFDetector)
				netFeatures:add(netFOrientation)
			end
		else
			if lossDescriptor == 1 then
				local netFDescriptor = define_netFDescriptor(stride)
				netFeatures:add(netFDetector)
				netFeatures:add(netFDescriptor)
			else
				netFeatures:add(netFDetector)
			end
		end
	end

	return netFeatures
end

function computeFeaturesDetector(stride, input)

	local netFDetector = define_netFDetector(stride)
	
	if opt.gpu == 1 then
		netFDetector:cuda()
	end

    local temp = netFDetector:forward(input)
    
    temp[temp:gt(0.5)] = 1
    temp[temp:le(0.5)] = 0

    return temp
end

function computeFeaturesOrientation(stride, input)

    local netFOrientation = define_netFOrientation(5)

    if opt.gpu == 1 then
    	netFOrientation:cuda()
    end

    local featuresOrientation = netFOrientation:forward(input)

    return featuresOrientation
end

function computeFeaturesDescriptor(stride, input)

    local netFDescriptor = define_netFDescriptor(stride)

    if opt.gpu == 1 then
    	netFDescriptor:cuda()
    end

    local featuresDescriptor = netFDescriptor:forward(input)
    
    featuresDescriptor[featuresDescriptor:gt(0.5)] = 1
    featuresDescriptor[featuresDescriptor:le(0.5)] = 0

    return featuresDescriptor
end

function computeFeatures(lossDetector, lossOrientation, lossDescriptor, stride, input)
    
    local features = nil

    if lossDetector == 1 then
        local featuresDetector = computeFeaturesDetector(stride, input)
        if lossOrientation == 1 then
            local featuresOrientation = computeFeaturesOrientation(stride, input)
            if lossDescriptor == 1 then
                local featuresDescriptor = computeFeaturesDescriptor(stride, input)
                features = torch.cat(torch.cat(featuresDetector, featuresOrientation, 2), featuresDescriptor, 2)
            else
                features = torch.cat(featuresDetector, featuresOrientation, 2)
            end
        else
            if lossDescriptor == 1 then
                local featuresDescriptor = computeFeaturesDescriptor(stride, input)
                features = torch.cat(featuresDetector, featuresDescriptor, 2)
            else
                features = featuresDetector:clone()
            end
        end
    end

    return features
end

function computeDetectorWeights(lossDetector, lossOrientation, lossDescriptor, inputFeatures, targetFeatures)

    local tFeat = targetFeatures[{{},{1},{},{}}]:float()
    local iFeat = inputFeatures[{{},{1},{},{}}]:float()

    local weights = torch.zeros(tFeat:size())
    local valFeatures = tFeat:numel() / tFeat[tFeat:gt(0.5)]:numel()
    local valBackground = tFeat:numel() / tFeat[tFeat:le(0.5)]:numel()

    --weights[tFeat:ge(0.5)] = valFeatures
    --weights[tFeat:lt(0.5)] = valBackground
    weights[torch.cmin(tFeat:ge(0.5), iFeat:lt(0.5))] = valFeatures
    weights[torch.cmin(tFeat:lt(0.5), iFeat:ge(0.5))] = valBackground

    return weights
end

function computeDescriptorWeights(lossDetector, lossOrientation, lossDescriptor, inputFeatures, targetFeatures)
    
    local tFeat = targetFeatures[{{},{lossDetector + lossOrientation*3 + 1, lossDetector + lossOrientation*3 + 256},{},{}}]:float()
    local iFeat = inputFeatures[{{},{lossDetector + lossOrientation*3 + 1, lossDetector + lossOrientation*3 + 256},{},{}}]:float()

    local weights = torch.zeros(tFeat:size())
    
    weights[torch.cmin(tFeat:ge(0.5), iFeat:lt(0.5))] = 1
    weights[torch.cmin(tFeat:lt(0.5), iFeat:ge(0.5))] = 1

    return weights
end

function computeOrientationWeights(lossDetector, lossOrientation, lossDescriptor, inputFeatures, targetFeatures)
    
    local output_map = inputFeatures[{{},{1},{},{}}]:clone()
	output_map[output_map:gt(0.5)] = 1
    output_map[output_map:le(0.5)] = 0
    local target_map = targetFeatures[{{},{1},{},{}}]:clone()
	target_map[target_map:gt(0.5)] = 1
    target_map[target_map:le(0.5)] = 0
	local features_map = torch.add(target_map[{{},{1},{},{}}], output_map[{{},{1},{},{}}])
	features_map[features_map:gt(1)] = 1
	local weights = torch.cat(torch.cat(features_map, features_map, 2), features_map, 2)

    return weights
end

function define_netNoise(output_gan_nc)

	local SRM1 = torch.Tensor({{0, 0, 0, 0, 0}, {0, -1, 2, -1, 0}, {0, 2, -4, 2, 0}, {0, -1, 2, -1, 0}, {0, 0, 0, 0, 0}}):mul(0.25)
	local SRM2 = torch.Tensor({{-1, 2, -2, 2, -1}, {2, -6, 8, -6, 2}, {-2, 8, -12, 8, -2}, {2, -6, 8, -6, 2}, {-1, 2, -2, 2, -1}}):mul(1/12)
	local SRM3 = torch.Tensor({{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 1, -2, 1, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}):mul(0.5)

	local SRM = torch.Tensor(3,output_gan_nc,5,5)
	SRM[{{1},{1},{},{}}] = SRM1
	SRM[{{2},{1},{},{}}] = SRM2
	SRM[{{3},{1},{},{}}] = SRM3

	if output_gan_nc == 3 then
		SRM[{{1},{2},{},{}}] = SRM1
		SRM[{{2},{2},{},{}}] = SRM2
		SRM[{{3},{2},{},{}}] = SRM3
		SRM[{{1},{3},{},{}}] = SRM1
		SRM[{{2},{3},{},{}}] = SRM2
		SRM[{{3},{3},{},{}}] = SRM3
	end

	local net = nn.Sequential()
	local conv = nn.SpatialConvolution(output_gan_nc,3,5,5,1,1,2,2)
	conv.weight = SRM
	conv.bias:zero()

	net:add(conv)

	return net
end

function define_RGB2Gray()

	local  kernel = torch.Tensor(1, 3, 1, 1)
	kernel[{{1},{1},{1},{1}}] = 0.299
	kernel[{{1},{2},{1},{1}}] = 0.587
	kernel[{{1},{3},{1},{1}}] = 0.114

	local net = nn.Sequential()
	local conv = nn.SpatialConvolution(3,1,1,1,1,1)
	conv.weight = kernel
	conv.bias:zero()

	net:add(conv)

	return net
end