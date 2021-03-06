# Ducument-scanning
108學年第二學期
# 影像處理與實習 - 期末專題報告
## 摘要：
市面上目前有許多手機掃描APP，如Adobe Scan、Office Lens、HP Smart……等，透過拍照的方式就能產生如同掃描的效果，於是本專題想藉由研究此類APP原理，實踐其功能。在研究的過程中發現上述每一款APP都有兩個致命的缺點，其一(下左圖)為當光源不均時，掃描出影像中的有些字會太淺，導致閱讀困難，其二(下右圖)為若於拍攝時，紙張有弧度，則無法使文字平整。不過礙於時間不足，本專題僅就第一個問題作改良。
  |光源不均|紙張不平整|
  |-----------|------------|
  |![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/problem%20of%20uneven%20illumination.PNG)|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/problem%20of%20unflatting%20paper.PNG)| 
## 一、方法與流程圖
![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/flow%20chart.svg)
灰色框部分皆為運用課本內容中的指令，紅色框部分則為自己寫的程式，因此僅就紅色框部分的程式及特別的部份做說明。
1. 利用邊界交點找四個頂點  
(1) 找出上下左右皆為白色的點即為邊界交點  
(2) 將原圖形上下左右等分(四等分)，每塊裡取一個與原圖形邊界點最近的點


2. 修正光源不均、將字以外的部分都變成白色(去除背景格線)
  * 原本有兩個想法  
  (1) 找出白紙中最亮的點，及白紙中最暗的點，依據線性的方式，將值調整成光均勻時的情況  
  => 線性不一定符合真實情況  
  (2) 直接用適應性閥值化的方式  
  => 只有黑與白兩種顏色，並非灰階，無法凸顯鉛筆的部份，下圖為使用適應性閥值化的結果。
  
  |平均法(Mean)|高斯法(Gauss)|
  |-----------|------------|
  |![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/Adaptive%20Thresholding(Gaussian).jpg)|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/Adaptive%20Thresholding(Mean).jpg)|  

  * 改良後的想法：利用適應性閥值的原理  
  (1)	找出遮罩(blur)中最亮的點，即白紙的部份  
  (2)	計算最亮點與真實白色(255)的誤差  
  (3)	將區域內的所有值加上誤差  
  (4)	將偏白色的部份(>=225)變成白色  

3.	影像增強  
原本使用的是Beta校正，但調整到最後發現轉換的input-output圖與Gamma校正的十分類似，因此便改用計算較為簡便的Gamma校正。

|Beta校正a=1.2, b=0.7|Gamma校正γ=2.0|
|-------------------|--------------|
|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/Beta%E6%A0%A1%E6%AD%A3%E8%BD%89%E6%8F%9B%E5%87%BD%E6%95%B8.jpeg)|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/Gamma%E6%A0%A1%E6%AD%A3%E8%BD%89%E6%8F%9B%E5%87%BD%E6%95%B8.jpeg)|

## 二、實驗結果：(以0001.jpg為範例)

|原圖(Original Image)|Cammy邊緣偵測(Canny Edge Detection)|霍夫直線轉換(Hough Line Detection)|利用邊界找到四個頂點(Get Cornor)|
|-------------------|----------------------------------|--------------------------------|--------------------------|
|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/example/0001/Original%20Image.jpg)|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/example/0001/Canny%20Edge%20Detection.jpg)|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/example/0001/Hough%20Line%20Detection.jpg)|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/example/0001/Get%20Cornor.jpg)|

|仿射轉換(Perspective Transform)|光源修正(Correction)|Gamma校正(完成圖)(gamma_correction(final))|使用Adobe Scan的結果|
|-------------------|----------------------------------|--------------------------------|----------------------------------|
|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/example/0001/Perspective%20Transform.jpg)|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/example/0001/Correction.jpg)|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/example/0001/gamma_correction(final).jpg)|![image](https://github.com/Sunnie0101/Ducument-scanning/blob/main/img/example/0001/Adobe%20Scan.jpg)|


**可以發現紅色圓圈中文字變淺的部份得到了改善**
### 程式限制：
1. 當紙張有弧度時無法使用(霍夫直線轉換找不到直線)
2. 每張圖最適合霍夫直線轉換的線長度參數(紅色框部分)不一樣，
	如0001.jpg適合325，而0002.jpg適合300  
 `lines = cv2.HoughLines(edges, 1, math.pi/180.0, 325)  
 #0001:325  0002:300`
 
## 三、貢獻說明：
以課本所學實現目前市面上的掃描APP，並且修正因光線不均勻所造成的字體過淺(詳細於「方法與流程圖」中說明)。

## 四、參考文獻：
1.	Python List sort()方法
https://www.runoob.com/python/att-list-sort.html
2.	A4纸 – 百度百科
https://baike.baidu.com/item/A4%E7%BA%B8
張元翔，數位影像處理：Python程式實作，全華圖書，2019。















