function oof(fname, radius, outfile)
	img = double(loadtif(fname));

	% Zoom images
 %    outsz = round(size(img) * 0.7);
 %    [y x z]= ndgrid(linspace(1, size(img, 1), outsz(1)),...
	%                 linspace(1, size(img, 2), outsz(2)),...
	%                 linspace(1, size(img, 3), outsz(3)));
	% img = interp3(img, x, y, z);

	oof = oof3response(img, radius);
    % oof = padarray(oof, [18, 18, 18]);

    if exist(fname, 'file') == 2
        delete(outfile);
    end

	for K=1:size(oof,3)
		imwrite(uint8(imrotate(oof(:,:,K), 90) * 100), outfile, 'WriteMode', 'append');
	end
end

