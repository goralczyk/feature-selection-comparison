function selected_features = Afshar2020_filter(data, n_genes)
% Programmed by Majid Afshar on 15/08/19
% Ph.D. Candidate
% Department of Computer Science
% Memorial University of Newfoundland
% mman23 [AT] mun [DOT] ca 

% Programmed by Hamid Usefi
% Associate Professor of Mathematics
% Department of Mathematics and Statistics
% Memorial University of Newfoundland
% usefi [AT] mun [DOT] ca | www.math.mun.ca/~usefi/

% Input: A dataset
% Output: Selected features and the resulting classification accuracy and also standard deviation of # of selected features and accuracies

warning off;
%%================================Data Path=====================================
userName = char(java.lang.System.getProperty('user.name'));
OS = computer;

switch(OS)
    case 'GLNXA64'
        dir = 'home';
        
    case 'MACI64'
        dir = 'Users';
        
    case 'PCWIN64'
        dir = 'Users';
end

path = ['/', dir,'/', userName, '/Documents/datasets/'];

%%=============================Parameters==================================

clusters=n_genes; %size of k
runIter = 1; %number of independent runs
t = 50;
    %==========================Reading Dataset=============================
    
    data = data(randperm(size(data, 1)), :);
    
    orgData = data;

    %==========================Initialization=============================
    
    featuresPicked=zeros(clusters, runIter);
    eliteCluster = zeros(runIter, 4);
    max_acc=zeros(runIter, 2);
    F=cell(runIter);
   %==========================Data Prepration=============================
        data = data(randperm(size(data, 1)), :);
        
        [r, c] = size(data);
        allF = c - 1;

        %==============================Variables===========================
        A = data(:,1:end-1);
        C=A;
        B = data(:,end);

        %=========================Irrelevant removal=======================
        iA = pinv(A);
        X = iA * B;
        outliersLen=allF;
        listX=zeros(5, length(X));
        cleanedF=zeros(5, length(X));
        listX(1,:)=X;
        ii=1;
        while outliersLen>(allF*.021)
            outliers = isoutlier(abs(listX(ii,1:outliersLen)),'mean');
            tmp=find(outliers==1);
            outliersLen=length(tmp);
            cleanedF(ii,1:outliersLen)=tmp;
            ii=ii+1;
            listX(ii,1:outliersLen)=listX(ii-1,tmp);
        end
        if outliersLen<10
            TF = isoutlier(abs(X),'mean');
            outliersLen=sum(TF);
        end
        uniqueX=sort(unique(abs(X)),'descend');
        threshold=mean(uniqueX(1:outliersLen*10));
        cleanedF=1:allF;      
        while length(cleanedF)>(outliersLen*(2/(ii-1)))
            irrF = find(abs(X) < threshold);
            irrF = [irrF, X(irrF)];
            irrF = sortrows(irrF, 2, 'descend');
            irrF = irrF(:, 1);
            threshold=threshold*1.03;
            cleanedF = setxor([1:allF], irrF);
        end
        A=A(:,cleanedF');
        C=C(:,cleanedF');
        allF = length(cleanedF);
        
        %========================Perturbantion matrix======================
        svdA = svd(A);
        smallestAan = min(svdA);
        iA = pinv(A);
        X = iA * B;
        minPer = min(A) .* 10^-3 .* smallestAan;
        maxPer = max(A) .* 10^-2 .* smallestAan;
        s=3;
        mError =10^(-s)* smallestAan;
        perVal=zeros(r, allF);
        px=zeros(r, allF);
        parfor z=1:t
            perVal=mError.*rand(r, allF,1);
            nr=norm(perVal);
            pA = A + perVal;
            piA = pinv(pA);
            Xtilda=piA * B;
            DX = abs(Xtilda - X);
            px(z, :)=DX;
        end
        
        pX = mean(px)';
        ent=real(-nansum(C'.*log(C'),2));
   %====================Sorting PX and Finding top ranked features====================
        pX=smooth(pX,'sgolay');
        uniquePX=length(unique(pX));
        roundMetric=20;
        roundedPX=pX;
        while uniquePX> 50
            roundedPX=round(pX,roundMetric);
            roundMetric=roundMetric-1;
            uniquePX=length(unique(roundedPX));
        end
        out=[];
        uniquekeys=unique(roundedPX);
        indexOut=1;
        for key=1:uniquePX
           selectedPX=find(roundedPX==uniquekeys(key)); 
           filteredEnt=ent(selectedPX);
           uniqueEnt=length(unique(filteredEnt));
           roundMetric=5;
           roundedEnt=filteredEnt;
           while uniqueEnt> 20
               roundedEnt=round(filteredEnt,roundMetric);
               roundMetric=roundMetric-1;
               uniqueEnt=length(unique(roundedEnt));
           end
           lenEnt=length(unique(roundedEnt));
           uniqueEnt=unique(roundedEnt);
           for index=1:lenEnt
              selectedEnt=find(roundedEnt==uniqueEnt(index));
              filteredX=X(selectedPX(selectedEnt));
              rankFilteredX=[selectedPX(selectedEnt),filteredX];
              sortedFilteredX=sortrows( rankFilteredX, 2, 'descend');
              out(indexOut)=sortedFilteredX(1,1);
              indexOut=indexOut+1;
           end
        end
        outEnt=ent(out');
        rankOutEnt=[out',outEnt];
        sortedRankOutEnt=sortrows( rankOutEnt, 2, 'ascend');
        sortedRankOutEnt=[(1:length(sortedRankOutEnt(:,1)))' sortedRankOutEnt];
        outX=X(out');
        rankOutX=[out',outX];
        sortedRankOutX=sortrows( rankOutX, 2, 'descend');
        sortedRankOutX=[(1:length(sortedRankOutX(:,1)))' sortedRankOutX];
        for i=1:length(sortedRankOutX(:,1))
            indexX=find(sortedRankOutEnt(i,2)==sortedRankOutX(:,2));
             sortedRankOutEnt(i,4)=indexX+sortedRankOutEnt(i,1);  
        end
        sortedRankOutEnt=sortrows( sortedRankOutEnt, 4, 'ascend');
        out=cleanedF(sortedRankOutEnt(:,2));
        upperBand = min(clusters, length(out));
       %===========================Classifiying============================
        selected_features = out' - 1
