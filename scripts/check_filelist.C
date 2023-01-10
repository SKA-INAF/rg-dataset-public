
std::string ExtractSubString(const std::string& s, const std::string& pattern, bool extractleft=true)
{
	size_t pos = s.find(pattern);
	if(pos==std::string::npos) return s;
	if(extractleft) return s.substr(0,pos);
	else return s.substr(pos+1,s.length()-pos);
}

std::string ExtractFileNameFromPath(const std::string& s,bool strip_extension=false)
{
	char sep = '/';
	size_t pos = s.rfind(sep, s.length());
	std::string filename= s;
  if (pos != string::npos) {
  	filename= s.substr(pos+1, s.length() - pos);
  }
	if(strip_extension) {
		std::string filename_noext= ExtractSubString(filename,".");
		filename= filename_noext;
	}
  return filename;
}

int check_filelist(std::string filename="filelist.txt")
{
	TTree* data= new TTree;
	data->ReadFile(filename.c_str(),"filename_img/C:filename_reg/C",',');
	
	char filename_img[4096];
	char filename_reg[4096];

	data->SetBranchAddress("filename_img", filename_img);
	data->SetBranchAddress("filename_reg", filename_reg);

	cout<<"Reading file "<<filename<<" ..."<<endl;

	int N= 0;

	for(int i=0;i<data->GetEntries();i++)
	{
		data->GetEntry(i);
	
		std::string imgname= std::string(filename_img);
		std::string regname= std::string(filename_reg);
		std::string imgname_noext= ExtractFileNameFromPath(imgname, true);
		std::string regname_noext= ExtractFileNameFromPath(regname, true);

		if(imgname_noext!=regname_noext){
			cout<<imgname<<","<<regname<<endl;
			N++;
		}

	}//end loop filelist
	
	if(N>0) cout<<"#"<<N<<" wrong filelists!"<<endl;

	return 0;

}//close macro
