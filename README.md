# GENETIC-ALGORITHM

Thuật Toán Di Truyền (Genetic Algorithms)
- Genetic Algorithms (GA) là một phương pháp tối ưu hóa mạnh mẽ dựa trên nguyên lý chọn lọc tự nhiên, cho phép giải quyết các bài toán có và không có ràng buộc. Phương pháp này cho phép tìm kiếm giải pháp tối ưu trong không gian giải pháp rộng lớn mà không cần phải thử nghiệm từng kết hợp một cách thủ công. GA được phát triển bởi John H. Holland cùng với các sinh viên và đồng nghiệp tại Đại học Michigan, trong đó David E. Goldberg là một trong những nhân vật nổi bật.
- Cơ chế hoạt động: GA bắt đầu với một nhóm giải pháp ban đầu. Ở mỗi bước, thuật toán sẽ chọn các các cá thể từ quần thể có giải pháp tốt nhất để làm “cha mẹ”, sau đó lai tạo và đột biến để tạo ra “con cái” cho thế hệ tiếp theo. Qua nhiều thế hệ, nhóm giải pháp này dần dần cải thiện và tiến tới giải pháp tối ưu.


Giả sử rằng, một công ty đang sở hữu một chiếc xe tải chuyên dụng để vận chuyển hàng hóa. Mỗi sản phẩm mà công ty dự định chất lên xe tải không chỉ có lợi nhuận tiềm năng khác nhau mà còn chiếm những không gian khác nhau trên xe. 
- Vấn đề đặt ra là:  Làm thế nào để tối ưu hóa việc chất hàng lên xe tải nhằm đạt được lợi nhuận cao nhất, đồng thời phải tuân thủ các giới hạn về không gian và trọng lượng của xe ?
-  Chính vì vậy, để giải quyết bài toán này, chúng em sẽ áp dụng thuật toán di truyền (Genetic Algorithm), một phương pháp tối ưu hóa mạnh mẽ có khả năng tìm kiếm trong không gian giải pháp rộng lớn một cách hiệu quả.

# KẾT QUẢ KHÔNG GIAN 3D CỦA THÙNG XE SAU KHI CHẠY XONG GIẢI THUẬT

  ![image](https://github.com/user-attachments/assets/bde21b4c-294b-4ea5-8f35-3ebbdd65955c)


1.  Cấu trúc của hệ thống sản phẩm 

![image](https://github.com/user-attachments/assets/46535a27-af5d-4abb-8e80-141d5e8561d2)

2.	Thông Tin Về Sản Phẩm và Các Giới Hạn Vận Chuyển
- Trong quá trình thử nghiệm, sẽ có tổng cộng 14 các mặt hànG. Không phải tất cả chúng đều vừa với xe tải. Vì:
  + Trọng lượng tối đa của xe tải: 150 kg
  + Tổng trọng lượng của 14 mặt hàng : 329.9 kg > 150 kg  (vượt quá sức chứa tối đa của thùng xe)

![image](https://github.com/user-attachments/assets/83e83d60-716d-48e6-9220-4df6ba89bf6c)

3. Cơ sở lý thuyết (các bước tiến hành giải thuật)

![image](https://github.com/user-attachments/assets/f3aea53b-25fd-4d8f-aa0f-c78423c5591f)

Sơ đồ biểu diễn cấu trúc của thuật toán di truyền

![image](https://github.com/user-attachments/assets/cc45161b-b31d-4546-b27d-6344c3838835)
