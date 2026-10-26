# Optional installed-R comparison; R is not a Python runtime dependency.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L) stop('Usage: Rscript tools/export_mashr_reference.R output.json')
stopifnot(as.character(utils::packageVersion('mashr'))=='0.2.79')
suppressPackageStartupMessages(library(mashr))
B<-rbind(c(1,-.5),c(.2,1.3),c(-.8,-.4),c(.1,.2),c(1.4,1.1),c(-1.2,.8))
S<-rbind(c(.5,.7),c(.6,.4),c(.8,.5),c(.4,.6),c(.7,.7),c(.6,.8))
V<-matrix(c(1,.2,.2,1),2)
U<-list(independent=diag(2),shared=matrix(1,2,2),specific=diag(c(1,0)))
make_case<-function(name,alpha=0,fixed=FALSE,penalty=1,grid=1) {
 data<-mash_set_data(B,S,alpha=alpha,V=V)
 if(fixed){
  g<-list(pi=c(.15,.25,.35,.25),Ulist=U,grid=1,usepointmass=TRUE)
  m<-mash(data,g=g,fixg=TRUE,verbose=FALSE,pi_thresh=0,outputlevel=3,output_lfdr=TRUE)
 }else{
  m<-mash(data,Ulist=U,grid=grid,normalizeU=FALSE,usepointmass=TRUE,
          prior=if(penalty==1)'uniform' else 'nullbiased',nullweight=penalty,
          verbose=FALSE,pi_thresh=0,outputlevel=3,output_lfdr=TRUE)
 }
 list(name=name,effects=unname(B),standard_errors=unname(S),correlation=unname(V),
      alpha=alpha,fixed=fixed,null_weight=penalty,grid=as.list(grid),patterns=lapply(U,unname),
      weights=as.list(as.numeric(m$fitted_g$pi)),posterior_mean=unname(ashr::get_pm(m)),
      posterior_sd=unname(ashr::get_psd(m)),lfsr=unname(ashr::get_lfsr(m)),
      lfdr=unname(ashr::get_lfdr(m)),
      posterior_covariance=lapply(seq_len(nrow(B)),function(i)unname(m$result$PosteriorCov[,,i])))
}
data<-mash_set_data(B,S,V=V)
record<-list(reference=list(package='mashr',version=as.character(packageVersion('mashr')),
 R=R.version.string,namespace=getNamespaceInfo(asNamespace('mashr'),'path')),
 cases=list(make_case('EE-fixed',fixed=TRUE),make_case('EZ-fixed',alpha=1,fixed=TRUE),
            make_case('EE-estimated'),make_case('EZ-estimated',alpha=1),
            make_case('EE-null-biased',penalty=10),make_case('EE-scales',grid=c(.5,1,2))),
 pca=list(patterns=lapply(cov_pca(data,2),unname)))
jsonlite::write_json(record,args[1],auto_unbox=TRUE,digits=17,pretty=TRUE)
