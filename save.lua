
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