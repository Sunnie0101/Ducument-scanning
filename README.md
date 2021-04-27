# Ducument-scanning
108學年第二學期
# 影像處理與實習 - 期末專題報告
## 摘要：
市面上目前有許多手機掃描APP，如Adobe Scan、Office Lens、HP Smart……等，透過拍照的方式就能產生如同掃描的效果，於是本專題想藉由研究此類APP原理，實踐其功能。在研究的過程中發現上述每一款APP都有兩個致命的缺點，其一(上圖)為當光源不均時，掃描出影像中的有些字會太淺，導致閱讀困難，其二(下圖)為若於拍攝時，紙張有弧度，則無法使文字平整。不過礙於時間不足，本專題僅就第一個問題作改良。
## 一、方法與流程圖

灰色框部分皆為運用課本內容中的指令，紅色框部分則為自己寫的程式，因此僅就紅色框部分的程式及特別的部份做說明。
1. 利用邊界交點找四個頂點  
(1) 找出上下左右皆為白色的點即為邊界交點  
(2) 將原圖形上下左右等分(四等分)，如右圖，每塊裡取一個與原圖形邊界點最近的點


2. 修正光源不均、將字以外的部分都變成白色(去除背景格線)
  * 原本有兩個想法  
  (1) 找出白紙中最亮的點，及白紙中最暗的點，依據線性的方式，將值調整成光均勻時的情況  
  => 線性不一定符合真實情況  
  (2) 直接用適應性閥值化的方式  
  => 只有黑與白兩種顏色，並非灰階，無法凸顯鉛筆的部份，下圖為使用適應性閥值化的結果。
  
  |平均法(Mean)|高斯法(Gauss)|
  |-----------|------------|
  |![image](https://user-images.githubusercontent.com/60318542/116192975-37840f80-a761-11eb-910e-8e5c9a66c19c.jpeg)|![image](https://user-images.githubusercontent.com/60318542/116192993-3fdc4a80-a761-11eb-9195-64b83061e06c.jpeg)|  

  * 改良後的想法：利用適應性閥值的原理  
  (1)	找出遮罩(blur)中最亮的點，即白紙的部份  
  (2)	計算最亮點與真實白色(255)的誤差  
  (3)	將區域內的所有值加上誤差  
  (4)	將偏白色的部份(>=225)變成白色  



