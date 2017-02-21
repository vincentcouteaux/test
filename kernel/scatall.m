addpath('../../../Downloads/scatnet-0.2')
addpath_scatnet

X = csvread('Xtr.csv');
X = X(:, 1:3072);
X = (X - min(min(X)))/(max(max(X))-min(min(X)));
%x = reshape(X(12, 1:1024), [32, 32])';
[Wop,filters] = wavelet_factory_2d(size(x));
%[S,U] = scat(x,Wop);
%[sc_table, meta] = format_scat(S);
scat_m1 = zeros(5000, 32*4*4*3);
for k = 1:5000
  disp(k)
  x = reshape(X(12, 1:1024), [32, 32])';
  [S,~] = scat(x,Wop);
  [sc_table, meta] = format_scat(S);
  scat_m1(k, 1:512) = sc_table(meta.order == 1, :, :)(:)';
  x = reshape(X(12, 1025:2048), [32, 32])';
  [S,~] = scat(x,Wop);
  [sc_table, meta] = format_scat(S);
  scat_m1(k, 513:1024) = sc_table(meta.order == 1, :, :)(:)';
  x = reshape(X(12, 2049:3072), [32, 32])';
  [S,~] = scat(x,Wop);
  [sc_table, meta] = format_scat(S);
  scat_m1(k, 1025:1536) = sc_table(meta.order == 1, :, :)(:)';
end
csvwrite('Xtr_scat_m1.csv', scat_m1)