# Phân tích cảm xúc đánh giá sản phẩm Tiki bằng Machine Learning

## 1. Giới thiệu

Dự án thực hiện phân tích cảm xúc (Sentiment Analysis) trên dữ liệu đánh giá sản phẩm từ nền tảng thương mại điện tử Tiki, sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (Natural Language Processing – NLP) và học máy (Machine Learning).

Mục tiêu là xây dựng các mô hình có khả năng phân loại nội dung đánh giá của khách hàng thành ba nhóm cảm xúc: **Tích cực (Positive), Trung tính (Neutral) và Tiêu cực (Negative)**.

Dự án tập trung vào toàn bộ quy trình khai phá dữ liệu văn bản, từ khám phá dữ liệu, tiền xử lý, trực quan hóa, trích xuất đặc trưng đến xây dựng, đánh giá và cải thiện mô hình phân loại.

## 2. Mục tiêu dự án

- Khám phá cấu trúc, đặc điểm và chất lượng dữ liệu đánh giá sản phẩm.
- Tiền xử lý và làm sạch văn bản tiếng Việt phục vụ bài toán phân tích cảm xúc.
- Trực quan hóa phân phối nhãn cảm xúc và đặc điểm nội dung đánh giá.
- Trích xuất đặc trưng văn bản bằng TF-IDF.
- Xây dựng và so sánh hiệu quả của nhiều thuật toán học máy.
- Xử lý vấn đề mất cân bằng dữ liệu giữa các lớp cảm xúc.
- Đánh giá mô hình bằng nhiều chỉ số để phân tích khả năng nhận diện từng nhóm cảm xúc.

## 3. Dữ liệu sử dụng

Dữ liệu đầu vào gồm các tệp văn bản `.txt` chứa thông tin đánh giá sản phẩm. Các tệp được tổng hợp và chuyển đổi thành DataFrame để thuận tiện cho quá trình xử lý.

Sau bước chuyển đổi ban đầu, bộ dữ liệu có **13.652 mẫu và 8 thuộc tính**.

| Thuộc tính | Mô tả |
|---|---|
| `product_id` | Mã sản phẩm |
| `customer_name` | Tên khách hàng |
| `rating` | Điểm đánh giá |
| `review_title` | Tiêu đề đánh giá |
| `helpful_votes` | Số lượt đánh giá hữu ích |
| `verified_purchase` | Thông tin xác nhận mua hàng |
| `review_date` | Ngày đánh giá |
| `review_text` | Nội dung đánh giá |

Dựa trên giá trị `rating`, dữ liệu được gán nhãn cảm xúc:

- **Positive:** Điểm đánh giá từ 80 trở lên.
- **Neutral:** Điểm đánh giá từ 50 đến dưới 80.
- **Negative:** Điểm đánh giá dưới 50.

Bộ dữ liệu có sự mất cân bằng đáng kể: lớp Positive chiếm khoảng 85,6%, Neutral chiếm 11,8% và Negative chiếm 2,6%. Đây là thách thức quan trọng trong quá trình huấn luyện và đánh giá mô hình.

*Lưu ý: Các ngưỡng gán nhãn trên được sử dụng theo logic trong notebook và không nhất thiết tương đương với cách phân loại sao đánh giá thông thường.*

## 4. Quy trình thực hiện

### 4.1. Khám phá và chuẩn bị dữ liệu

- Đọc và tổng hợp dữ liệu từ nhiều tệp văn bản.
- Xây dựng DataFrame với các thuộc tính tương ứng.
- Kiểm tra kích thước, kiểu dữ liệu và đặc điểm của tập dữ liệu.
- Thống kê số lượng mẫu theo từng nhãn cảm xúc.
- Phân tích độ dài đánh giá dựa trên số ký tự và số từ.

### 4.2. Tiền xử lý văn bản tiếng Việt

- Chuẩn hóa văn bản về chữ thường.
- Xử lý emoji và các biểu tượng cảm xúc bằng cách chuyển đổi thành những token đại diện.
- Loại bỏ URL, HTML và các ký tự không cần thiết.
- Sử dụng danh sách stopwords tiếng Việt để loại bỏ từ ít giá trị.
- Bảo vệ các từ phủ định và từ biểu đạt cảm xúc quan trọng.
- Bổ sung stopwords theo lĩnh vực thương mại điện tử để giảm các từ xuất hiện phổ biến nhưng ít khả năng phân biệt cảm xúc.
- Tạo cột văn bản đã làm sạch để phục vụ trích xuất đặc trưng.

### 4.3. Trực quan hóa dữ liệu

Sử dụng Matplotlib, Seaborn và WordCloud để:

- Trực quan hóa phân phối các lớp cảm xúc.
- So sánh độ dài văn bản giữa các nhóm cảm xúc.
- Khám phá các từ xuất hiện phổ biến trong từng nhóm Positive, Neutral và Negative.
- Nhận diện sự mất cân bằng dữ liệu và những đặc điểm có thể ảnh hưởng đến kết quả phân loại.

### 4.4. Trích xuất đặc trưng văn bản

Sử dụng **TF-IDF (Term Frequency–Inverse Document Frequency)** để chuyển đổi nội dung đánh giá thành các vector số, làm đầu vào cho các thuật toán học máy.

Notebook sử dụng `TfidfVectorizer` với giới hạn tối đa 10.000 đặc trưng và kết hợp unigram, bigram thông qua `ngram_range=(1, 2)`.

Dữ liệu được chia thành các tập huấn luyện, xác thực và kiểm tra nhằm hỗ trợ huấn luyện, lựa chọn cấu hình và đánh giá mô hình.

### 4.5. Xây dựng và đánh giá mô hình

Các thuật toán được triển khai và thử nghiệm trong notebook gồm:

- **Naive Bayes:** Thử nghiệm các biến thể Multinomial Naive Bayes và Complement Naive Bayes.
- **Logistic Regression:** Xây dựng mô hình cơ sở và thử nghiệm điều chỉnh tham số.
- **Linear Support Vector Machine (SVM):** Phân loại văn bản dựa trên siêu phẳng tuyến tính.
- **Random Forest:** Khảo sát mô hình học tổ hợp dựa trên nhiều cây quyết định.
- **Stacking Classifier:** Kết hợp nhiều mô hình phân loại nhằm khai thác ưu điểm của từng mô hình.

### 4.6. Cải thiện mô hình và xử lý mất cân bằng dữ liệu

Notebook khảo sát một số kỹ thuật nhằm tăng khả năng nhận diện các lớp thiểu số:

- **Class Weighting:** Sử dụng `class_weight='balanced'` để điều chỉnh trọng số giữa các lớp.
- **Ensemble Learning:** Thử nghiệm Stacking Classifier.
- **Threshold Tuning:** Điều chỉnh ngưỡng quyết định cho các lớp Negative và Neutral.
- **Oversampling:** Tăng số lượng mẫu thuộc lớp thiểu số trên tập huấn luyện.

Các kỹ thuật được đánh giá dựa trên khả năng cải thiện kết quả phân loại, đặc biệt là đối với các lớp Negative và Neutral, đồng thời hạn chế rò rỉ dữ liệu giữa tập huấn luyện và tập kiểm tra.

## 5. Chỉ số đánh giá

Các chỉ số được sử dụng để đánh giá mô hình gồm:

| Chỉ số | Ý nghĩa |
|---|---|
| Accuracy | Tỷ lệ dự đoán đúng trên toàn bộ mẫu |
| Precision | Tỷ lệ dự đoán đúng trong số các mẫu được dự đoán thuộc một lớp |
| Recall | Tỷ lệ mẫu thuộc một lớp được mô hình nhận diện đúng |
| F1-score | Trung bình điều hòa giữa Precision và Recall |
| Macro F1-score | Trung bình F1-score của các lớp với trọng số ngang nhau |
| Confusion Matrix | Ma trận thể hiện số lượng dự đoán đúng và nhầm lẫn giữa các lớp |

Do dữ liệu mất cân bằng, **Macro F1-score và Recall của từng lớp** là những chỉ số quan trọng để đánh giá khả năng nhận diện các nhóm cảm xúc thiểu số, thay vì chỉ dựa vào Accuracy.

## 6. Công nghệ và thư viện

- **Ngôn ngữ:** Python
- **Môi trường:** Google Colab, Jupyter Notebook
- **Xử lý dữ liệu:** Pandas, NumPy
- **Trực quan hóa:** Matplotlib, Seaborn, WordCloud
- **Học máy:** Scikit-learn
- **Xử lý văn bản:** TF-IDF, Regular Expressions (`re`)
- **Xử lý dữ liệu ma trận thưa:** SciPy

## 7. Hướng dẫn chạy dự án

### Cách 1: Chạy trên Google Colab

1. Mở file `Khai_pha_du_lieu.ipynb` bằng Google Colab.
2. Chuẩn bị tệp dữ liệu nén `tiki_dataset.zip`.
3. Chuẩn bị file stopwords tiếng Việt `vietnamese-stopwords-dash.txt`.
4. Tải các tệp cần thiết lên Colab khi notebook yêu cầu.
5. Chạy lần lượt các cell theo thứ tự từ đầu đến cuối.
6. Theo dõi các biểu đồ, báo cáo phân loại và kết quả so sánh mô hình.

### Cách 2: Chạy trên máy tính cá nhân

1. Tải hoặc clone repository về máy.
2. Cài đặt Python và các thư viện cần thiết.
3. Đặt các tệp dữ liệu vào đúng đường dẫn mà notebook sử dụng hoặc chỉnh sửa đường dẫn tương ứng.
4. Mở notebook bằng Jupyter Notebook, JupyterLab hoặc môi trường hỗ trợ Jupyter.
5. Chạy lần lượt các cell.

Cài đặt các thư viện chính bằng lệnh:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud scipy jupyter
```

## 8. Kết quả và nhận xét

Qua quá trình thực nghiệm, dự án cho thấy một số vấn đề đáng chú ý:

- Dữ liệu đánh giá có sự mất cân bằng rõ rệt giữa các lớp cảm xúc.
- Các mô hình có thể đạt Accuracy cao nhưng vẫn nhận diện hạn chế đối với lớp Negative và Neutral.
- Kỹ thuật `class_weight='balanced'` giúp cải thiện khả năng nhận diện các lớp thiểu số trong những thử nghiệm trên Logistic Regression và SVM.
- Stacking cải thiện một số kết quả tổng thể nhưng chưa chắc mang lại Macro F1-score tốt nhất.
- Điều chỉnh ngưỡng dự đoán và oversampling là những hướng tiếp cận được khảo sát nhằm cải thiện hiệu quả phân loại.

Kết quả cuối cùng cần được đối chiếu với báo cáo thực nghiệm trong notebook trước khi xác định mô hình tốt nhất.

## 9. Kiến thức và kỹ năng đạt được

Thông qua dự án, tôi có cơ hội thực hành:

- Khám phá và tiền xử lý dữ liệu thực tế.
- Phân tích và trực quan hóa dữ liệu văn bản.
- Tiền xử lý văn bản tiếng Việt cho bài toán NLP.
- Trích xuất đặc trưng bằng TF-IDF.
- Xây dựng, huấn luyện và so sánh các mô hình học máy.
- Đánh giá mô hình bằng nhiều chỉ số phân loại.
- Xử lý mất cân bằng dữ liệu và phân tích sự đánh đổi giữa các chỉ số.
- Trình bày kết quả thực nghiệm và rút ra nhận xét từ dữ liệu.

## 10. Hướng phát triển

- Thử nghiệm các phương pháp tách từ tiếng Việt chuyên biệt.
- Khảo sát thêm những kỹ thuật xử lý mất cân bằng dữ liệu.
- Tối ưu siêu tham số có hệ thống dựa trên tập xác thực.
- So sánh với các mô hình ngôn ngữ tiền huấn luyện dành cho tiếng Việt.
- Chuẩn hóa quy trình thực nghiệm để tăng khả năng tái lập kết quả.

## Tác giả

**Vũ Khánh Vân**

GitHub: [Vukhanhvan09](https://github.com/Vukhanhvan09)

---

*Dự án được thực hiện trong quá trình học tập và thực hành môn Khai phá dữ liệu, tập trung vào phân tích cảm xúc, xử lý ngôn ngữ tự nhiên và học máy.*
