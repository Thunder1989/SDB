num = size((coefMatInCells),2);
for i = 1:num
    HeatMap(cell2mat(coefMatInCells(1,i)))
end