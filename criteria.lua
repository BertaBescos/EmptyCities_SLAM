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