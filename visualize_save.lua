require 'image'
util = paths.dofile('util/util.lua')


function display()
	-- display
	
	if opt.counter % opt.display_freq == 0 and opt.display then	
		print('visualize')
		createRealFake()
		local img_input = util.scale_batch(realGray_A:float(),100,100*aspect_ratio):add(1):div(2)
		if opt.input_gan_nc == 3 then
			img_input = util.deprocess_batch(img_input)
		end
		disp.image(img_input, {win=opt.display_id, title=opt.name .. ' input'})
		local mask_input = util.scale_batch(real_C:float(),100,100*aspect_ratio):add(1):div(2)
		disp.image(mask_input, {win=opt.display_id+1, title=opt.name .. ' mask'})
		local img_output = util.scale_batch(fake_B:float(),100,100*aspect_ratio):add(1):div(2)
		if opt.output_gan_nc == 3 then
			img_input = util.deprocess_batch(img_output)
		end
		disp.image(img_output, {win=opt.display_id+2, title=opt.name .. ' output'})
		local img_target = util.scale_batch(realGray_B:float(),100,100*aspect_ratio):add(1):div(2)
		if opt.output_gan_nc == 3 then
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
end

function save_display()
	if opt.counter % opt.save_display_freq == 0 and opt.display then
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
					if opt.input_gan_nc == 1 then 
						image_out = torch.cat(realGray_A[i2]:float():add(1):div(2), fake_B[i2]:float():add(1):div(2), 3)
					else
						image_out = torch.cat(util.deprocess(realGray_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3)
					end
				else
					if opt.input_gan_nc == 1 then
						image_out = torch.cat(image_out, torch.cat(realGray_A[i2]:float():add(1):div(2), fake_B[i2]:float():add(1):div(2),3), 2)
					else
						image_out = torch.cat(image_out, torch.cat(util.deprocess(realGray_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3), 2)
					end
				end
			end
		end
		image.save(paths.concat(opt.checkpoints_dir,  opt.name , opt.counter .. '_train_res.png'), image_out)
		opt.serial_batches=serial_batches
	end
end

function val_display()
	if (opt.counter % opt.val_display_freq == 0 or opt.counter == 1) and opt.display then
		val_createRealFake()
		val_errL1 = criterionGenerator:forward(val_fake_B, val_realGray_B)
		local img_input = util.scale_batch(val_realGray_A:float(),100,100*aspect_ratio):add(1):div(2)
		if opt.input_gan_nc == 3 then
			img_input = util.deprocess_batch(img_input)
		end
		disp.image(img_input, {win=opt.display_id+20, title=opt.name .. ' val_input'})
		local mask_input = util.scale_batch(val_real_C:float(),100,100*aspect_ratio):add(1):div(2)
		disp.image(mask_input, {win=opt.display_id+21, title=opt.name .. ' val_mask'})
		local img_output = util.scale_batch(val_fake_B:float(),100,100*aspect_ratio):add(1):div(2)
		if opt.output_gan_nc == 3 then
			img_output = util.deprocess_batch(img_output)
		end
		disp.image(img_output, {win=opt.display_id+22, title=opt.name .. ' val_output'})
		local img_target = util.scale_batch(val_realGray_B:float(),100,100*aspect_ratio):add(1):div(2)
		if opt.output_gan_nc == 3 then
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
end

function display_plot(epoch, i)
	if opt.counter % opt.print_freq == 0 then
		local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errL1=errL1 and errL1 or -1, errFeatures=errFeatures and errFeatures or -1, errERFNet=errERFNet and errERFNet or -1, val_errL1=val_errL1 and val_errL1 or -1}
		local curItInBatch = ((i-1) / opt.batchSize)
		local totalItInBatch = math.floor(math.min(synth_data_size, opt.ntrain) / opt.batchSize)
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
end

function save_latest_model()
	-- save latest model
	if opt.counter % opt.save_latest_freq == 0 then
		print(('saving the latest model (epoch %d, iters %d)'):format(epoch, opt.counter))
		torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
		torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
		if opt.NSYNTH_DATA_ROOT ~= '' then
			torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_SS.net'), netSS:clearState())
		end
	end
end

function save_epoch_model(epoch)
	if epoch % opt.save_epoch_freq == 0 then
		torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
		torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
		if opt.NSYNTH_DATA_ROOT ~= '' then
			torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_SS.net'), netSS:clearState())
		end
	end
end

function save_options()
	-- save opt
	file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
	file:writeObject(opt)
	file:close()
end