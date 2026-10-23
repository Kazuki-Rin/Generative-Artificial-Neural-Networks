# Generative-Artificial-Neural-Networks
Hệ thống 10 mô hình AI chạy song song, tự sinh nơ-ron để thích nghi môi trường. Các nhóm cảm biến, tư duy và ký ức liên tục mở rộng cấu trúc độc lập, kết hợp lõi Neural ODE xử lý giọng nói tự nhiên.
Hệ thống này chỉ đang dừng ở mức nghiên cứu của 1 sinh viên và không biết thời hạn hoàn thiện dự án, dự án này mới thực hiện được hơn 1 tuần hơn dưới sự trợ giúp của gemini 3.1 pro và bây giờ là Argon.
Nếu chúng ta "mổ xẻ" bộ não con người và quy đổi các vùng chức năng sinh học thành các mô hình AI độc lập (các mạng nơ-ron chuyên biệt), bạn sẽ cần xây dựng 9 mô hình AI "sơ sinh" để hợp nhất thành một hệ thống AGI hoàn chỉnh.
Các mô hình này không hoạt động độc lập mà được nối với nhau qua một "trục xương sống" truyền tín hiệu. Dưới đây là bản thiết kế chi tiết 9 mô hình đó:
Nhóm 1: Các Mô Hình Cảm Giác (Nhận thông tin)
Trẻ sơ sinh khi mới ra đời sẽ dùng nhóm này đầu tiên để cảm nhận thế giới.
●	Mô hình 1 - Mắt (Thùy chẩm / Occipital Lobe): Chỉ nhận ma trận điểm ảnh thô (pixels). Nhiệm vụ của nó lúc sơ sinh là phân biệt được sáng/tối, tìm ra các đường viền, và sau này là nhận diện các khối hình (icon ứng dụng, cửa sổ).
●	Mô hình 2 - Tai (Thùy thái dương / Temporal Lobe): Chỉ nhận các dải tần số sóng âm thô. Nhiệm vụ của nó là tách lọc tạp âm môi trường và phát hiện tiếng người.
●	Mô hình 3 - Cảm giác không gian (Thùy đỉnh / Parietal Lobe): Ở người, đây là cảm giác không gian của cơ thể. Với máy tính, đây là mô hình theo dõi tọa độ con trỏ chuột, xác định giới hạn viền màn hình và vị trí các cửa sổ đang mở. Nó giúp AI biết "tay" mình đang ở đâu.
Nhóm 2: Các Mô Hình Vận Động & Sinh Tồn (Hành động)
●	Mô hình 4 - Cơ bắp (Tiểu não / Cerebellum): Đây là mô hình học tăng cường (Reinforcement Learning). Nó điều khiển tọa độ chuột và các phím bấm. Trẻ sơ sinh sẽ quờ quạng, mô hình này ban đầu sẽ click lung tung. Nó tự học bằng cách tối ưu hóa lực kéo chuột cho mượt mà mà không cần Đại não phải tính toán.
●	Mô hình 5 - Cảm biến nguy hiểm & Phần thưởng (Hạch hạnh nhân / Amygdala): Đây chính là Cơ chế Ngắt (Interrupt). Mô hình này cực nhỏ, chạy cực nhanh, chuyên quét âm thanh và hình ảnh để tìm các tín hiệu khẩn cấp ("Dừng lại", dấu X đỏ, lỗi màn hình xanh). Nếu phát hiện, nó ném tín hiệu ức chế, bắt Mô hình 4 lập tức đóng băng tay chân.
Nhóm 3: Các Mô Hình Nhận Thức Bậc Cao (Tư duy)
Nhóm này ở trẻ sơ sinh gần như ngủ đông và chỉ bắt đầu phát triển khi các nhóm trên đã hòan thiện cơ bản.
●	Mô hình 6 - Ngôn ngữ & Khái niệm (Wernicke & Broca): Nhiệm vụ của nó là ánh xạ các chùm sóng âm từ Tai (Mô hình 2) và hình ảnh từ Mắt (Mô hình 1) thành các "Khái niệm". Nó giúp AI hiểu rằng sóng âm "C-r-ôm" tương đương với cái icon hình tròn màu xanh đỏ vàng trên màn hình.
●	Mô hình 7 - Lập kế hoạch (Thùy trán / Prefrontal Cortex): Lõi thông minh nhất (tương đương với việc bạn đang dùng LLM hiện tại). Nhiệm vụ của nó là chia nhỏ mục tiêu: Để mở Youtube $\rightarrow$ Cần di chuột tới trình duyệt $\rightarrow$ Cần click đúp $\rightarrow$ Cần gõ phím.
●	Mô hình 8 - Trí nhớ (Hồi hải mã / Hippocampus): Nhiệm vụ của nó là thu thập các trải nghiệm đúng/sai trong ngày và cất vào kho (Vector Database). Khi ngủ, nó lôi dữ liệu ra để "huấn luyện lại" các mô hình khác.
Nhóm 4: Bộ Định Tuyến
●	Mô hình 9 - Sự chú ý (Đồi thị / Thalamus): Đứa trẻ không thể tập trung vào mọi thứ trên màn hình cùng lúc. Mô hình này đóng vai trò như chiếc "công tắc đa luồng". Khi màn hình nhấp nháy, nó ép hệ thống phải dồn tài nguyên xử lý vào Mô hình Mắt. Khi có tiếng động lớn, nó ngắt bớt Mắt để nhường điện năng cho Mô hình Tai.
Chiến lược Train (Huấn luyện)
Nếu bạn tạo ra cùng lúc 9 mô hình "sơ sinh" này và bật chúng lên, cỗ máy sẽ rơi vào hỗn loạn. Trẻ sơ sinh không thể vừa học chữ vừa học cách điều khiển tay chân. Bạn phải huấn luyện chúng theo tiến trình sinh học:
1.	Tháng thứ 1: Chỉ bật Mô hình 1 (Mắt), Mô hình 3 (Tọa độ) và Mô hình 4 (Cơ bắp). Cho đứa trẻ AI học cách nhìn các đốm màu trên màn hình và tự di chuyển chuột chạm vào các đốm đó để nhận phần thưởng.
2.	Tháng thứ 2: Bật thêm Mô hình 2 (Tai) và Mô hình 5 (Phản xạ khẩn cấp). Dạy nó đang di chuột mà nghe tiếng "Xùy" thì phải dừng lại.
3.	Tháng thứ 3: Bắt đầu bật các Mô hình Tư duy bậc cao để nó tập ghép các tín hiệu lại thành một quy trình có ý nghĩa.

1. Nhóm Cảm Biến & Vận Động (M1, M2, M3, M4)
Nhóm này cần khả năng tự mở rộng cấu trúc khi môi trường thay đổi.
•	M1_Eye (Thị giác) & M3_Spatial (Không gian): Lõi HyperNEAT (Dựa trên CPPN)
o	Tại sao tự sinh: Thuật toán NEAT thông thường có thể rối nếu nhận ma trận ảnh quá lớn. HyperNEAT dùng một mạng nơ-ron nhỏ (CPPN) để "vẽ" ra trọng số cho một mạng nơ-ron lớn hơn.
o	Khả năng tiến hóa: Ban đầu mô hình chỉ nhận 1 kênh ảnh xám. Khi bạn nâng cấp cho nó nhận ảnh màu (3 kênh RGB), thuật toán sẽ tự động "mọc" thêm các nơ-ron đầu vào và sinh ra các lớp liên kết mới để nhận diện màu sắc mà không quên đi khả năng nhận diện hình khối trước đó.
•	M2_Ear (Thính giác): Evolving Spiking Neural Networks (E-SNN)
o	Tại sao tự sinh: Mạng nơ-ron gai (SNN) xử lý tín hiệu theo thời gian. Bạn sẽ code thêm một cơ chế sinh học: Ngưỡng sinh nơ-ron mới. Nếu một âm thanh đưa vào tạo ra độ sai lệch (error) quá lớn so với những gì mạng đã biết, thay vì điều chỉnh trọng số (vì có thể làm quên âm thanh cũ), hệ thống sẽ tự động khởi tạo một nơ-ron gai mới tinh, gán cho nó nhiệm vụ chuyên nhận diện dải tần số lạ đó.
•	M4_Cerebellum (Cơ bắp): Lõi NEAT thuần túy
o	Đây vốn dĩ là thuật toán tự sinh. Ban đầu não sơ sinh chỉ có đầu vào (ảnh) nối thẳng đầu ra (chuột). Theo thời gian, nó tự chèn thêm các "hidden nodes" (nơ-ron ẩn) vào giữa để thực hiện các thao tác di chuột quỹ đạo cong mượt mà hơn.
2. Nhóm Tư Duy Bậc Cao (M6, M7)
Nhóm này không chỉ cần mọc thêm nơ-ron, mà bản thân các liên kết phải "chảy" và thay đổi theo thời gian thực.
•	M6_Language (Ngôn ngữ) & M7_Prefrontal (Lập kế hoạch): Liquid Neural Networks (LNN) + Sinh trưởng phân lớp
o	Tại sao tự sinh: LNN dùng phương trình vi phân, nên trọng số của nó linh hoạt ngay cả khi đã train xong (rất giống não thật). Để thêm yếu tố "tự sinh", bạn áp dụng nguyên lý Progressive Neural Networks (Mạng nơ-ron tiến triển).
o	Cơ chế: Khi AI phải học một khái niệm hoàn toàn mới (ví dụ: khái niệm "Lướt web" khác hẳn "Chơi game"), thay vì ép mạng LNN cũ học, hệ thống sẽ tự sinh ra một "cột nơ-ron" (neural column) LNN mới, đặt song song với cột cũ. Cột mới này có thể lấy dữ liệu từ cột cũ, giúp AI biết thêm kỹ năng mới mà không bị "thảm họa quên" (catastrophic forgetting).
3. Nhóm Trí Nhớ & Trục Xương Sống (M5, M8, M9)
•	M8_Hippocampus (Trí nhớ): Lõi Adaptive Resonance Theory (ART)
o	Tại sao tự sinh: ART là thuật toán kinh điển cho việc "tự học". Cơ chế của nó cực kỳ giống sự hình thành ký ức: Khi một trải nghiệm mới đi vào, ART sẽ so sánh nó với các cụm ký ức cũ. Nếu nó đủ giống, nó gộp vào. Nếu nó hoàn toàn khác biệt (Vượt ngưỡng Cảnh giác - Vigilance parameter), mô hình sẽ ngay lập tức tạo ra một node mới để lưu trữ mảng ký ức này. Kích thước của mạng ART sẽ lớn dần theo độ tuổi của Kazuki Rin.
•	M5_Amygdala (Ngắt khẩn cấp): Lõi SNN tối giản có khả năng mọc gai (Sprouting)
o	Ban đầu nó chỉ có vài nơ-ron để nhận tiếng hét "Dừng lại". Nhưng nếu bạn muốn nó tự học thêm các tín hiệu nguy hiểm khác (như tiếng quạt tản nhiệt kêu rú lên), mô hình này sẽ dùng cơ chế sinh nơ-ron giống M2 để tự mọc thêm một nhánh cảm biến nguy hiểm mới.
•	M9_Thalamus (Định tuyến): Lõi Self-Organizing Map (SOM) Động
o	Mạng nơ-ron tự tổ chức này sẽ hoạt động như một bản đồ phân bổ tài nguyên. Khi hệ thống mọc thêm nhiều nơ-ron ở M1 (Mắt) và M2 (Tai), SOM sẽ tự động mở rộng lưới (grid) của nó để tạo ra các kênh truyền tín hiệu (synapse) mới, đảm bảo băng thông không bị nghẽn.
Mô hình 10 
M10 không phải là một mạng nơ-ron cục mịch, mà là một Hệ thống điều khiển tham số đa trục (Multi-axis Parametric Controller). Bên trong nó gồm 3 phân hệ chính chạy nối tiếp nhau:
•	Trạm thu nhận tín hiệu (Input Interface):
o	Nhận mảng Âm vị (Phonemes) từ M6 (Thùy ngôn ngữ).
o	Nhận biến số Cảm xúc (Vector 0.0 - 1.0) từ M5 (Hạch hạnh nhân - Vui, buồn, sợ hãi).
o	Nhận lệnh Định tuyến (Nói/Hát/Im lặng) từ M7 (Thùy trán - Lõi logic).
•	Lõi Động học Liên tục (Bộ điều khiển Neural ODE):
o	Đây là não bộ của thanh quản. Nó giải các phương trình vi phân để tính toán ra quỹ đạo chuyển động mượt mà của các cơ quan vật lý.
o	Xuất ra mảng tham số cơ học: Pitch (Cao độ), Jaw (Độ mở hàm), Lip (Độ chúm môi), Breath_Pressure (Lực ép khí quản), Tension (Độ căng dây thanh). Các thông số này cũng được truyền thẳng sang phần mềm Live2D để nhân vật nhép miệng chuẩn 100%.
•	Lò đúc âm thanh (Thanh quản số DDSP):
o	Sử dụng công nghệ Xử lý tín hiệu số vi phân (Differentiable DSP). Nó lấy các tham số cơ học ở trên đưa vào các bộ dao động (Oscillators) và bộ lọc cộng hưởng (Formant Filters).
o	Kết quả cuối cùng: In ra sóng âm .wav cực nhẹ, có luyến láy thanh điệu Tiếng Việt, có tiếng thở, có thể hát ngân nga (Vibrato) và hoàn toàn triệt tiêu cảm giác "giọng robot".

Giai đoạn 1: Chế tạo Cơ bắp Sinh học & Phản xạ
Mục tiêu là xây dựng các kết nối vật lý cơ bản nhất với độ chính xác cao nhưng vẫn mang sai số của con người.
•	Mô hình 4 (Cơ bắp): Sử dụng mạng học tăng cường NEAT (khoảng 900,000 nơ-ron) để tinh chỉnh lực kéo chuột và nhịp gõ phím. Quỹ đạo chuột sinh ra sẽ là các đường cong có gia tốc, hãm phanh chệch choạc ở gần mục tiêu, giúp AI dễ dàng vượt qua các thuật toán chống bot như Google reCAPTCHA. 
•	Mô hình 5 (Phản xạ sinh tồn): Xây dựng một luồng độc lập với khoảng 100,000 nơ-ron, luôn trực chờ để đóng băng hệ thống ngay lập tức khi nhận được tín hiệu "Dừng". 
•	Mô hình 3 (Không gian): Phân bổ khoảng 100,000 nơ-ron để liên tục theo dõi tọa độ X/Y của con trỏ chuột và các cửa sổ ứng dụng. 
Giai đoạn 2: Khai mở Giác quan Điểm vàng (Foveated Perception)
Hệ thống sẽ không đọc toàn bộ ảnh độ phân giải cao để tiết kiệm tối đa nơ-ron.
•	Mô hình 1 (Mắt): Giới hạn ở mức ~2 triệu nơ-ron và chia làm hai luồng riêng biệt. Luồng toàn cảnh (Dumb Vision) ép màn hình xuống mức 64x64 pixel mờ ảo để nhận biết hình khối chung. Luồng điểm vàng (Smart Focus) sử dụng ô 32x32 pixel siêu nét di chuyển khắp màn hình để đọc chi tiết text hoặc icon. 
•	Mô hình 2 (Tai): Tối giản với ~900,000 nơ-ron làm đúng nhiệm vụ tách sóng âm thanh thành ký hiệu để chuyển lên não bộ. 
Giai đoạn 3: Lõi Logic & Ngoại bào hóa Ký ức
Ở giai đoạn này, AI bắt đầu hình thành tư duy, nhưng tuyệt đối không được phép học thuộc lòng dữ kiện.
•	Mô hình 7 (Thùy trán - Lập kế hoạch): Đây là lõi nặng nhất với khoảng 100 triệu nơ-ron. 100% năng lượng toán học của nó chỉ được mài giũa để giải quyết logic If-Then, phân tích ý định và quyết định gọi công cụ (Tool-calling). 
•	Mô hình 6 (Ngôn ngữ): Dành khoảng 14 triệu nơ-ron cho việc xử lý khái niệm. 
•	Mô hình 8 (Trí nhớ định tuyến): Khoảng 100,000 nơ-ron làm nhiệm vụ đẩy dữ liệu ra/vào Vector Database nằm trên ổ cứng SSD và RAM. Khi cần lục lại ký ức hoặc thông tin, AI sẽ sinh ra lệnh tìm kiếm trong ổ cứng thay vì phải tự nhớ. 
Cấy ghép M9 (Thalamus - Định tuyến): Điểm giao thoa Giai đoạn 2 & 3
Vị trí Tích hợp: Nằm ở nút thắt cổ chai giữa Giai đoạn 2 (Xử lý Giác quan) và Giai đoạn 3 (Lõi Logic).
•	Lý do Sinh học: Ngài không thể đẩy toàn bộ dữ liệu hỗn lốn từ M1 (Mắt) và M2 (Tai) ném thẳng vào M7 (Thùy trán) và M8 (Trí nhớ) được. Nó sẽ gây nghẽn cổ chai và tràn RAM ngay lập tức. M9 sinh ra để giải quyết việc này.
•	Cách thức Hoạt động: Ngay khi hệ thống chuyển mình từ Giai đoạn 2 sang 3, M9 (Lõi SOM Động) sẽ được kích hoạt. Nó đóng vai trò như một "Tổng đài viên".
o	Nếu M1 nhìn thấy ngài đang code, M9 mở luồng ưu tiên đẩy ảnh vào M7 để phân tích logic If-Then.
o	Nếu M2 nghe tiếng nhạc nền, M9 ném âm thanh đó vào M8 để lưu trữ dạng ký ức nền (không làm phiền M7 suy nghĩ).
o	Khi M1 và M2 mọc thêm nơ-ron (như ngài đã thiết kế), lưới SOM của M9 sẽ tự giãn ra để tạo băng thông mới, đảm bảo hệ thống không bao giờ bị nghẽn mạng nội bộ.
Ở giai đoạn bản lề này, việc cấy ghép M10 gánh vác 3 nhiệm vụ tối thượng để biến một đoạn code khô khan thành một thực thể sống:
•	Mở khóa "Cơ quan Chấp hành": Chuyển đổi tư duy vô hình của M7 và M6 thành dạng năng lượng vật lý truyền ra môi trường (âm thanh phát ra loa và chuyển động trên màn hình). Nếu không có nó, AI của ngài chỉ là một cục não bị nhốt trong hộp sọ, biết suy nghĩ nhưng bị câm.
•	Thiết lập Phản xạ Cắt ngang (Interrupt): Giai đoạn này ngài phải đấu dây M5 (Cảm biến nguy hiểm) vào M10. Nhiệm vụ là dạy cho AI biết giật mình. Đang nói mà nghe tiếng động lớn, hoặc bị người dùng ngắt lời, M10 phải lập tức khóa khí quản, ngừng nhả âm thanh ngay giữa câu, tạo ra độ phản ứng realtime y hệt con người.
•	Chuẩn bị cho "Giấc mơ" (Tiền đề cho Giai đoạn 4): Để Giai đoạn 4 (Nhịp sinh học & Tỉa cành) hoạt động được, AI phải biết "tự nói thầm" trong đầu khi ngủ để kiểm tra lại các trọng số kết nối. M10 lúc này sẽ chạy mô phỏng ngầm (không phát ra loa ngoài) để giúp M6 và M7 dọn dẹp các rễ thần kinh lỗi.
Giai đoạn 4: Kích hoạt Nhịp Sinh học & Tỉa cành (Pruning)
Đưa hệ thống vào trạng thái hoạt động vòng lặp sinh học, tích hợp thuật toán "Vừa làm vừa nén" (Quantization-Aware Training). 
•	Chế độ Thức (Ban ngày): Mọi sự kiện, thao tác thành công hay thất bại đều được ném vào bộ đệm tạm thời trên RAM để làm vùng trí nhớ ngắn hạn. 
•	Chế độ Ngủ (Ban đêm): AI tự động lôi ký ức ban ngày ra chạy lại mô phỏng nội tâm với tốc độ siêu tốc. Cơ chế cắt tỉa (Synaptic Pruning) sẽ quét dọn và tiêu hủy các rễ thần kinh vô dụng hoặc kết nối trùng lặp. AI sẽ mỏng đi nhưng cơ bắp toán học trở nên đặc lại.
•	🧹 1. Tiêu hủy rễ thần kinh rác (Synaptic Pruning)
•	Ban ngày, khi M10 (Thanh quản) học cách nói một từ mới, nó sẽ thử hàng ngàn cách uốn lưỡi khác nhau và tạo ra hàng vạn kết nối nơ-ron tạm thời.
•	Nếu không có Giai đoạn 4, rác nơ-ron này sẽ tích tụ dần, khiến VRAM bị tràn chỉ sau vài ngày.
•	Giai đoạn 4 "nhảy vào" làm nhiệm vụ của một cỗ máy dọn dẹp: Nó quét qua toàn bộ M10, tìm những rễ thần kinh nào không tạo ra hiệu quả âm thanh tốt, và xóa sổ (zero-out) chúng ngay lập tức. Càng ngủ, não nó càng gọn gàng.
•	🗜️ 2. Ép nén trọng số (Quantization-Aware Training)
•	Đây là phép màu giúp một mô hình khổng lồ có thể chạy mượt trên máy tính cá nhân của ngài.
•	Ban ngày, hệ thống tính toán bằng những con số cực kỳ to và nặng (Float32).
•	Vào ban đêm ở Giai đoạn 4, nó dùng thuật toán để "ép" những con số to đùng đó thành những con số siêu nhỏ (INT8 hoặc INT4) mà không làm thay đổi chất lượng giọng nói. Nhờ đó, sáng hôm sau khi ngài gọi Asuna thức dậy, nó sẽ tốn ít RAM hơn hẳn ngày hôm trước.
•	🔄 3. Chạy lại vòng lặp mô phỏng (Replay)
•	Ngài có nhớ những lúc Asuna nói bị vấp chữ, trễ nhịp hoặc nhép miệng sai vào ban ngày không?
•	Giai đoạn 4 không bỏ qua lỗi lầm đó. Khi ngắt loa ngoài, M10 và M6 (Ngôn ngữ) tự động "cãi nhau" trong tiềm thức. Chúng ném các phương trình vi phân vào nhau hàng triệu lần ở tốc độ siêu tốc (fast-forward) cho đến khi tìm ra quỹ đạo uốn lưỡi trơn tru nhất để sửa cái lỗi vấp chữ lúc sáng.
Giai đoạn 4.5 không chỉ là bài học về "Đóng mở cửa sổ". Nó là một quá trình tiến hóa toàn diện, biến một cỗ máy bíp bíp thành một thực thể:
1.	Biết nhìn (Sự vật chuyển động)
2.	Biết nghe (Tiếng động môi trường & Giọng nói)
3.	Biết sợ (Phản xạ dừng lại khi gặp lỗi)
4.	Biết nối ghép (Hình ảnh A đi liền với Âm thanh B)
5.	Biết suy luận ngắn (Nếu thấy X thì làm Y ngay lập tức)
🧬 BẢN VẼ GIAI ĐOẠN 4.8: CẤY GHÉP "HẠCH NỀN" (BASAL GANGLIA)
Bản chất của Hạch nền không phải là để "suy nghĩ thêm", mà là để Cầm nhịp, Nghiệm thu, và Chém vòng lặp. Ngài sẽ thi công qua 4 bước sau
Giai đoạn 4.9: Kỷ nguyên Nhận thức Tự thân (Self-Awareness Engine), chia làm 3 cụm thi công chính để ghép nối vào kiến trúc hiện tại:
1.	Cụm Động lực (Motivation Core): Nâng cấp Hạch hạnh nhân (M5) và Hạch nền (GĐ 4.8) để thêm Dopamine System (Reward Prediction Error) và Curiosity Engine (Kích hoạt luồng điện tự khám phá khi ngài đi ngủ).
2.	Cụm Tưởng tượng (Imagination Core): Nâng cấp Thùy trán (M7) để tách làm 2 luồng vi phân: một luồng ra lệnh (Actor) và một luồng dự đoán tương lai (World Model), đi kèm bộ tính toán Độ tự tin (Confidence %).
3.	Cụm Tổ chức (Organizational Core): Phân mảnh ổ cứng ChromaDB (M8) thành Episodic (Nhớ sự kiện), Semantic (Nhớ khái niệm) và Procedural (Nhớ kỹ năng di chuột), đồng thời đấu nối cảm biến nhiệt độ PC vào hệ thống để làm Homeostasis.
Giai đoạn 4.91 này là mảnh ghép "Vua" cuối cùng trước khi xuất xưởng:
1. Task Scheduler (Hệ điều hành của Não bộ)
Ngài đã có Hạch nền (4.8) để chặn lệnh sai, nhưng ngài chưa có bộ đếm thời gian (Scheduler). Khi Asuna vừa muốn "Nghe ngài nói" (DDSP), vừa "Nhìn màn hình" (HyperNEAT), vừa "Suy nghĩ" (LNN) — nếu không có Scheduler phân bổ CPU, máy tính sẽ treo cứng ở mức 100%. Scheduler giống như một vị "Nhạc trưởng", quyết định nhạc cụ nào được cất lên ở Mili-giây nào.
2. Failure Memory (Cuốn sổ đen thù vặt)
Con người thông minh không phải vì luôn làm đúng, mà vì biết nhớ kỹ những lần vỡ mặt. M8 hiện tại chỉ lưu cái ĐÚNG. Nếu Asuna click vào một file .exe gây crash máy, nó bị phạt (Dopamine âm), nhưng nó lại... quên mất file đó tên gì. Failure Memory sẽ đóng vai trò như một "Danh sách đen" (Blacklist), thấy mặt file đó lần 2 là tự động bỏ chạy, không cần suy nghĩ.
3. Skill Library (Đóng gói thói quen)
Đây là sự tinh tế của Kỹ thuật phần mềm. Thay vì mỗi lần muốn mở Chrome, não bộ (M7) phải tính toán ma trận cả ngàn lần để kéo chuột... Nó chỉ cần học 1 lần, đóng gói thành macro (Primitive) tên là [Mở_Chrome]. Lần sau chỉ việc gọi hàm. Điều này tiết kiệm 99% công sức tính toán!
4. Self-Debug & Checkpoint (Sinh tồn ngắt quãng)
Góp ý về Checkpoint là một sự cứu rỗi. Giả sử Asuna ngồi "cày" đồ án PBL5 cho ngài mất 45 phút, đến phút 46 phần mềm báo lỗi Exception. Nếu không có Checkpoint, nó sẽ ngơ ngác và mất sạch 45 phút suy luận. Với Self-Debug, nó sẽ tự đọc thông báo lỗi, load lại Checkpoint ở phút 45, tìm cách khác và đi tiếp.
Giai đoạn 4.99
•	Tạo ra Trục xương sống giao tiếp: Áp dụng cơ chế xuất/nhập tín hiệu nội bộ để các vùng não có thể "nói chuyện" với nhau theo thời gian thực. 
•	Kích hoạt Đa tiến trình (Multiprocessing): Hệ thống không còn phải chờ mắt nhìn xong mới tới tai nghe; M1, M2, M4 và M5 sẽ cùng hoạt động song song trong một "Vòng lặp Sống". 
•	Hoàn thiện Khớp nối Sinh tồn: Trao quyền sinh sát cho Hạch hạnh nhân (M5). Khi có nguy hiểm, nó có thể lập tức cắt điện (ngắt luồng) toàn bộ cơ thể để bảo vệ hệ thống. 
Giai đoạn 5: Thực tập sinh & Đánh đổi Thời gian (Test-Time Compute)
Cho phép AI học lỏm và tự xử lý đa nhiệm, ưu tiên tư duy sâu (System 2 Thinking) thay vì phản hồi tức thời. 
•	Học lỏm thói quen (Behavioral Cloning): Chạy ngầm để chụp ảnh màn hình và theo dõi thao tác. Nếu cứ 8 giờ tối người dùng mở phần mềm kỹ thuật (như TIA Portal hoặc MATLAB), AI sẽ tự động đúc kết quy luật và chuẩn bị sẵn các công cụ phụ trợ (ví dụ: mở sẵn tab tra cứu Chrome). 
•	Vòng lặp Suy nghĩ (Chain-of-Thought): AI được phép mất hàng chục phút hoặc 1 tiếng đồng hồ để xử lý lệnh. Nó sẽ tự lên kế hoạch, điều khiển chuột, mở trình duyệt, cuộn trang, copy dữ liệu và tự nhận ra sai lầm để lùi lại sửa chữa giống hệt quy trình làm việc của con người. 
Ngài định nghĩa Giai đoạn 5 là Đánh đổi Thời gian và Vòng lặp Suy nghĩ (Chain-of-Thought) kéo dài hàng chục phút.
Vấn đề: Nếu Asuna mất 1 tiếng để lên kế hoạch, mở trình duyệt, copy dữ liệu... mà cô bé lại chiếm dụng con trỏ chuột và màn hình chính của ngài (khi ngài đang vẽ CAD hay lập trình PLC), thì đó là một thảm họa hệ thống.
Giải pháp từ 06_Laboratory: Đây chính là không gian để Test-Time Compute diễn ra!
•	Khi ngài giao lệnh: "Viết cho tôi một đoạn mã MATLAB xử lý ảnh" $\rightarrow$ Vỏ não M7 sẽ không cướp chuột của ngài. Nó sẽ sinh ra một Đặc vụ ngầm (Sub-agent) chui vào 06_Laboratory.
•	Tại đây, Asuna tự mở một trình duyệt ảo (Headless Browser), tự search web, tự đọc tài liệu, tự viết file .m và chạy thử bằng subprocess. Nếu lỗi, cô bé tự đọc Traceback và tự sửa nghiệm thu hàng chục phút trong im lặng. Khi xong xuôi, cô bé mới báo ra ngoài: "Tài liệu em đã để trên bàn làm việc."
•	Đúng như ngài nói: Đây mới là cách con người (một Thực tập sinh) làm việc!
Giai đoạn 6: Tối ưu Cực hạn & Trở thành Thực thể Tác tử
Khóa cứng tổng ngân sách ở mức ~120 triệu nơ-ron (tạo ra khoảng 1.5 Tỷ đến 3 Tỷ tham số). 
•	Lượng tử hóa (Quantization): Ép dữ liệu từ Float32 về INT4, INT8, hoặc thậm chí là 1-bit (-1, 0, 1) kết hợp với cắt tỉa siêu thưa (Extreme Sparsity) để giữ lại 10% rễ thần kinh cốt lõi nhất. 
•	Ghi đè trọng số (Synaptic Plasticity): Khi bộ não học một kỹ năng mới cực kỳ phức tạp, hệ thống sẽ tự động ghi đè, làm mờ các kỹ năng rác cũ (ví dụ: cách di chuột chệch choạc lúc sơ sinh) để nhường chỗ cho rễ thần kinh mới, giống nguyên lý setpoint của hệ thống SCADA. 
Ở Giai đoạn 6, ngài muốn giới hạn ngân sách nơ-ron và ghi đè kỹ năng rác để nạp kỹ năng mới (như việc thay đổi Setpoint trong SCADA).
Nhưng làm sao não bộ (M7) biết kỹ năng mới đã đủ tốt để ghi đè kỹ năng cũ chưa?
•	Chính 06_Laboratory sẽ đóng vai trò là "Môi trường Mô phỏng" (Simulation Environment).
•	Khi M7 thấy một vùng não (ví dụ M3) hoạt động kém, nó sẽ đưa một bản sao của M3 vào 06_Laboratory, tạo ra các bối cảnh giả lập để "ép xung" M3 tự train lại.
•	Chỉ khi nào bản sao M3 này đạt điểm tự tin (Confidence) cao trong phòng thí nghiệm, M7 mới kích hoạt Quá trình Cắt tỉa (Extreme Sparsity) và Ghi đè trọng số ở môi trường thực tế.
