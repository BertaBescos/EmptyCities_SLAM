
require 'image'
require 'torch'

print('Loading data augmentation paramaters...')

data_aug = {}

function data_aug.apply(imA,imB,imC)

-- FLIP
if torch.uniform() > 0.5 then
    imA, imB, imC = data_aug.imflip(imA,imB,imC)
end

-- ROTATION
if torch.uniform() > 0.5 then
    imA, imB, imC = data_aug.rotate(imA,imB,imC)
end

-- GAUSSIAN BLUR
if torch.uniform() > 0.5 then
    imA = data_aug.gaussian_blur(imA)
end

-- GAUSSIAN NOISE
if torch.uniform() > 0.5 then
    imA = data_aug.gaussian_noise(imA)
end

-- CONTRAST VARIATION
if torch.uniform() > 0.5 then
    imA = data_aug.contrast(imA)
end

-- BRIGHTNESS VARIATION
if torch.uniform() > 0.5 then
    imA = data_aug.brightness(imA)
end

-- SATURATION
if torch.uniform() > 0.5 then
    imA = data_aug.saturation(imA)
end

-- DROPOUT
--if torch.uniform() > 0.5 then
  --  data_aug.dropoutn(imA)
--end

return imA, imB, imC
end

function data_aug.rotate(imA,imB,imC)

    deg = torch.uniform(-30,30)

    imA = image.rotate(imA, deg * math.pi / 180, 'bilinear')
    imB = image.rotate(imB, deg * math.pi / 180, 'bilinear')
    imC = image.rotate(imC, deg * math.pi / 180, 'bilinear')

    deg = math.abs(deg)
    h = imA:size(2)
    w = imA:size(3)
    bb_w = w * math.cos(deg * math.pi / 180) + h * math.sin(deg * math.pi / 180)
    bb_h = w * math.sin(deg * math.pi / 180) + h * math.cos(deg * math.pi / 180)

    gamma = math.atan2(bb_w, bb_w)
    delta = math.pi - deg * math.pi / 180 - gamma

    d = w * math.cos(deg * math.pi / 180)
    a = d * math.sin(deg * math.pi / 180) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    w_new = bb_w - 2 * x
    h_new = bb_h - 2 * y

    imA = image.crop(imA, (w-w_new)/2, (h-h_new)/2, w_new + (w-w_new)/2, h_new + (h-h_new)/2)
    imB = image.crop(imB, (w-w_new)/2, (h-h_new)/2, w_new + (w-w_new)/2, h_new + (h-h_new)/2)
    imC = image.crop(imC, (w-w_new)/2, (h-h_new)/2, w_new + (w-w_new)/2, h_new + (h-h_new)/2)

    return imA, imB, imC
end

function data_aug.imflip(imA,imB,imC)
    imA = image.hflip(imA)
    imB = image.hflip(imB)
    imC = image.hflip(imC)

    return imA, imB, imC
end

function data_aug.gaussian_blur(imA)
    local width = math.ceil(torch.uniform(0,6))
    local height = width
    local sigma_horz=torch.uniform(0.05, 0.25)
    local sigma_vert = sigma_horz
    local gs = image.gaussian{amplitude=1, 
                                normalize=true, 
                                width=width, 
                                height=height, 
                                sigma_horz=sigma_horz, 
                                sigma_vert=sigma_vert}
    imA = image.convolve(imA,gs,'same')
    imA = imA:add(-imA:min())
    imA = imA:div(imA:max())
    return imA 
end

function data_aug.gaussian_noise(imA)
    local augNoise = torch.uniform(0,0.1)
    local gs = torch.randn(imA:size()):float()*augNoise
    imA = torch.add(imA, gs)
    imA = imA:add(-imA:min())
    imA = imA:div(imA:max())
    return imA 
end

function data_aug.brightness(imA)
    local gs
    local var = 1
    gs = gs or imA.new()
    gs:resizeAs(imA):zero()
    local alpha = 1.0 + torch.uniform(-var, var)
    imA = imA:mul(alpha):add(1 - alpha, gs)
    imA = imA:add(-imA:min())
    imA = imA:div(imA:max())
    return imA
end

function data_aug.contrast(imA)
    local gs
    local var = 1
    gs = gs or imA.new()
    gs = imA:clone()
    gs:fill(gs:mean())
    local alpha = 1.0 + torch.uniform(-var, var)
    imA = imA:mul(alpha):add(1 - alpha, gs)
    imA = imA:add(-imA:min())
    imA = imA:div(imA:max())
    return imA
end

function data_aug.saturation(imA)
    local gs
    local var = 1
    gs = gs or imA.new()
    gs = imA:clone()
    local alpha = 1.0 + torch.uniform(-var, var)
    imA = imA:mul(alpha):add(1 - alpha, gs)
    imA = imA:add(-imA:min())
    imA = imA:div(imA:max())
    return imA
end

function data_aug.dropout(imA)
    local height = math.ceil(torch.uniform(10, 80))
    local width = math.ceil(torch.uniform(10, 80))
    h1 = math.ceil(torch.uniform(1e-2, imA:size(2)-height))
    w1 = math.ceil(torch.uniform(1e-2, imA:size(3)-width))
    imA[{{1},{h1,h1+height},{w1,w1+width}}] = 0
    return imA
end

return data_aug
