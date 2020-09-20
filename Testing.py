from tensorflow.keras.preprocessing import image
import numpy as np

def predict(model, pathToImage):
    
    #test = pd.read_csv('C:\\Users\\Shreya\\Final project\\test\\test.csv')

    #test.head()
    #test.shape

    #test_image = []
    #for i in tqdm(range(test.shape[0])):
        #img = image.load_img('C:/Users/Shreya/Final project/test/'+test['id'][i].astype('str')+'.png', target_size=(28,28,1), color_mode="grayscale")
        #img = image.img_to_array(img)
        #img = img/255
        #test_image.append(img)
    #test = np.array(test_image)

    #prediction = model.predict_classes(test)

    #sample = pd.read_csv('C:\\Users\\Shreya\\Final project\\sample_submission_I5njJSF.csv')
    #sample['label'] = prediction
    #sample.to_csv('C:\\Users\\Shreya\\Final project\\Newsample_cnn.csv', header=True, index=False)
    
    output_image=[]
    img = image.load_img(pathToImage, target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    output_image.append(img)
    test1 = np.array(output_image)
    op = model.predict_classes(test1)
    return op