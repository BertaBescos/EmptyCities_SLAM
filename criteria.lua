require 'image'

local WeightedAbsCriterion, parent = torch.class('nn.WeightedAbsCriterion', 'nn.Criterion')

function WeightedAbsCriterion:__init(weights, sizeAverage)
	parent.__init(self)
	if sizeAverage ~= nil then
		self.sizeAverage = sizeAverage
	else
		self.sizeAverage = true
 	end
	if weights ~= nil then
		self.weights = weights
	end
end

function WeightedAbsCriterion:updateOutput(input, target)
   
   	local weights = self.weights

   	local output = nil
   	if weights ~= nil then
		output = torch.abs(torch.cmul(input - target, weights)):sum()
	else
		output = torch.abs(input - target):sum()
	end

	if self.sizeAverage then
		output = output / input:numel()
	end

	self.output = output
   
	return self.output
end

function WeightedAbsCriterion:updateGradInput(input, target)

	self.gradInput:resizeAs(input)
	local temp = torch.cdiv(input - target, torch.abs(input - target))
	temp[temp:ne(temp)] = 1

	local weights = self.weights

	local output = nil
   	if weights ~= nil then
		output = torch.cmul(weights, temp)
	else
		output = temp:clone()
	end

	if self.sizeAverage then
		output = output / input:numel()
	end

	self.gradInput = output

	return self.gradInput
end

local WeightedBCECriterion, parent = torch.class('nn.WeightedBCECriterion', 'nn.Criterion')

function WeightedBCECriterion:__init(weights, sizeAverage)
	parent.__init(self)
	if sizeAverage ~= nil then
		self.sizeAverage = sizeAverage
	else
		self.sizeAverage = true
 	end
	if weights ~= nil then
		self.weights = weights
	end
end

function WeightedBCECriterion:updateOutput(input, target)
   
   	local weights = self.weights

   	local output = nil
   	if weights ~= nil then
		output = (torch.cmul(weights, - torch.cmul(target, input:clone():clamp(0.000001, 0.999999):log()) - torch.cmul((1 - target), (1 - input:clone():clamp(0.000001, 0.999999)):log())):sum()
	else
		output = (- torch.cmul(target, input:clone():clamp(0.000001, 0.999999):log()) - torch.cmul((1 - target), (1 - input:clone():clamp(0.000001, 0.999999)):log())):sum()
	end

	if self.sizeAverage then
		output = output / input:numel()
	end

	self.output = output
   
	return self.output
end

function WeightedBCECriterion:updateGradInput(input, target)

	self.gradInput:resizeAs(input)
	local temp = -torch.cdiv(target, input) + torch.cdiv((1 - target), (1 - input))
	temp[temp:ne(temp)] = 1

	local weights = self.weights

	local output = nil
   	if weights ~= nil then
		output = torch.cmul(weights, temp)
	else
		output = temp:clone()
	end

	if self.sizeAverage then
		output = output / input:numel()
	end

	self.gradInput = output

	return self.gradInput
end

function WeightedCECriterion()
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
	
	return cudnn.SpatialCrossEntropyCriterion(classWeights)
end