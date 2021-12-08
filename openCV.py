import cv2  # importing cv2 liberary
cam = cv2.VideoCapture(0)
count = 0
rinub='rinub'

while True:
    ret, img = cam.read()

    cv2.imshow("Test", img)

    if not ret:
        break

    k = cv2.waitKey(1)
    print("Image "+str(count)+"saved")
    file = 'E:\\Rinub_Thesis_Project\\Rinub_Data\\Check_Account_Balance\\'+str(rinub)+str(count)+'.jpg'
    cv2.imwrite(file, img)
    count += 1
    rinub='rinub'
    if count == 100:
        break
cam.release
cv2.destroyAllWindows



        # from keras.preprocessing import image
        # import numpy as np
        # import keras
        # import matplotlib.pyplot as plt
        # model = keras.models.load_model(r"C:\Users\Asus TUF\OneDrive\Desktop\akash_model\model_dl.h5")
        # img = image.load_img(r"C:\Users\Asus TUF\OneDrive\Desktop\image\9.jpg", target_size=(224, 224))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # image = np.vstack([x])
        # classes = model.predict(image, batch_size=256)
        # probabilities = model.predict_proba(image, batch_size=256)
        # probabilities_formatted = list(
        #     map("{:.2f}%".format, probabilities[0]*100))

        # # print(probabilities_formatted)
        # if probabilities_formatted == ['100.00%', '0.00%']:
        #     print('The recognized image is Rinub')
        #     plt.imshow(img)
        #     break

        # elif probabilities_formatted == ['0.00%', '100.00%']:
        #     print('The recognized image is an Akash')
        #     plt.imshow(img)
        #     break
        # else:
        #     break
