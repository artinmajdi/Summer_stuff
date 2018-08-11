import os, json
import pandas as pd
import csv

def gene_expr(dirp):
    for cur_path, directories, files in os.walk(dirp, topdown=True):
        for file in files:
            #print (file)
            if not file.startswith('annotations') and not file.endswith('.parcel'):
                #print (file)
                if file.endswith('.gz'):
                    path =os.path.join(cur_path,file)
                    df = pd.read_csv(path,compression='gzip', sep ='\t',names=['GENE_EXPR','Values']).assign(file_name=os.path.basename(file))
                    list.append(df)
                    df=pd.concat(list)
                else:
                    path =os.path.join(cur_path,file)
                    df = pd.read_csv(path,sep ='\t',names=['GENE_EXPR','Values']).assign(file_name=os.path.basename(file))
                    list.append(df)
                    df=pd.concat(list)
    return df


        #df.to_csv('C:\\Docs\\Poorni\\New_NCI_GDC\\AdGland\\AdGland_gene.csv', sep=',')
def metadata(pathjson):
    for index, js in enumerate(json_files):
        json_file= open(os.path.join(pathjson, js))
        json_text = json.load(json_file)
        for item in json_text:
            file_name=item['file_name']
            case_id = item['associated_entities'][0]['case_id']
            entity_submitter_id = item['associated_entities'][0]['entity_submitter_id']
            fin.append(file_name)
            casid.append(case_id)
            entid.append(entity_submitter_id)
            file =pd.DataFrame({'file_name' :fin,'case_id':casid,'entity_submitter_id':entid})
        #file.to_csv('C:\\Docs\\Poorni\\TCGA\\Output\\metadata.csv',sep=',',index=False)
    return file


def bio(dpath):
    list1 =[]
    fields =['sample_submitter_id','case_id','case_submitter_id','sample_type_id']
    #df1= pd.DataFrame(columns=fields)
    #columns=['sample_submitter_id','case_id','case_submitter_id','sample_type_id']
    for cur_path, directories, files in os.walk(dpath, topdown=True):
        for file in files:
            #print(file)
            path =os.path.join(cur_path,file)
            df1 = pd.read_csv(path,sep ='\t',header=0)
            list1.append(df1)
            df1=pd.concat(list1)
            #print ('file found writing df')
            #df1.to_csv('C:\\Docs\\Poorni\\TCGA\\Output\\samplefile.csv', sep=',')'''
    return df1

if __name__ == "__main__":
    #gene_expr path
    dirpath = '/home/groot/Final/geneexpr/'
    df= pd.DataFrame(columns=['GENE_EXPR','Values'])
    list =[]
    file_name=[]
    fil =[]
    #metadata path
    path_to_json ='/home/groot/Final/Meta/'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    dat =pd.DataFrame(columns=['file_name', 'case_id', 'entity_submitter_id'])
    jdata = []
    #sample path
    dpath = '/home/groot/Final/bio/'
    df1= pd.DataFrame()
    #columns=['sample_submitter_id','case_id','case_submitter_id','sample_type_id']
    fin =[]
    casid =[]
    entid =[]
    d=pd.DataFrame(columns=['GENE_EXPR','Values'])
    met =pd.DataFrame(columns=['file_name', 'case_id', 'entity_submitter_id'])
    met =metadata(pathjson ='/home/groot/Final/Meta/')
    d=gene_expr(dirp='/home/groot/Final/geneexpr/')
    #print (d.head())
    #print (met.head())
    dff=pd.DataFrame(columns=['file_name', 'case_id', 'entity_submitter_id','GENE_EXPR','Values'])
    dff=pd.merge(d,met, on='file_name', how='left')
    #print (dff)
    dff2=pd.DataFrame(columns =['sample_submitter_id','case_id','case_submitter_id','sample_type_id'])
    dff2 =bio(dpath='/home/groot/Final/bio/')
    dfnew =pd.DataFrame()
    #print(dff2.head())
    final=pd.DataFrame(columns =['file_name','case_id','entity_submitter_id','GENE_EXPR','Values','sample_submitter_id','case_submitter_id','sample_type_id'])
    final =pd.merge(dff,dff2, on='case_id', how='left')
    dfnew=final.filter(['file_name','case_id','entity_submitter_id','GENE_EXPR','Values','sample_submitter_id','case_submitter_id','sample_type_id'],axis=1)
    #dfnew[(dfnew['sample_type_id']==10)]
    dfnew[(dfnew['sample_type_id']==1)].to_csv('/home/groot/Output/sample_type_id_1.csv', sep=',')
    dfnew[(dfnew['sample_type_id']==2)].to_csv('/home/groot/Output/sample_type_id_2.csv', sep=',')
    dfnew[(dfnew['sample_type_id']==10)].to_csv('/home/groot/Output/sample_type_id_10.csv', sep=',')
    dfnew[(dfnew['sample_type_id']==11)].to_csv('/home/groot/Output/sample_type_id_11.csv', sep=',')
