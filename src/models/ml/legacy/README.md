# Legacy — KHÔNG dùng để train/báo cáo

4 script ở đây lấy thẳng cột `category` làm nhãn cho mỗi dòng của `python.csv`, bỏ qua
cột `instance_type` (nhãn thật). Vì `python.csv` lưu dữ liệu ở dạng "exploded" (mỗi câu
lặp lại 1 lần/category), cách đọc này khiến cùng một câu xuất hiện 5 lần với 5 nhãn khác
nhau, 4/5 lần sai → model gần như không học được gì (micro-F1 ~0.01, hoặc suy biến dự
đoán mọi nhãn cho mọi mẫu).

