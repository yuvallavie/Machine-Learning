function classification = labeler(args,threshold)
if(sum(args) <= threshold)
    classification = 1;
else
    classification = -1;
end

