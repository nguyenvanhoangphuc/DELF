# Cấu trúc data đầu vào

data/oxford5k_images: Thư mục chứa danh sách các file ảnh

data/gnd_roxford5k.mat: File chứa nhãn có cấu trúc như sau: 

- có các keys: ['__header__', '__version__', '__globals__', 'imlist', 'gnd', 'qimlist', 'priorindex_db', 'priorindex_queries', 'user']

- chỉ sử dụng các keys: ['qimlist', 'imlist', 'gnd']

+ với key 'qimlist': danh sách các tên file của các ảnh query

+ với key 'imlist': danh sách các tên file của các ảnh index

+ với key 'gnd': danh sách các dict với mỗi dict có 4 key cần quan tâm như sau: 

* 'bbx': bounding box của ảnh query

* ['easy', 'hard', 'junk']: lần lược là chỉ số trong tập index các ảnh liên quan nhất với query, các ảnh liên quan xa với query, các ảnh bỏ qua không đánh giá cho query.
