<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Confirm</title>
    <style>
        body {
            font-family: 'Times New Roman', Times, serif;
        }

        .outer-box {
            width: 30%;
            margin: 10% auto 0 auto;
            padding: 3rem;
        }

        .outer-box,
        .label,
        input[type=text],
        select,
        button,
        textarea#address {
            border: 2px solid #658baf;
        }

        .label {
            display: inline-block;
            width: 20%;
            padding: 7px;
            margin-bottom: 4%;
            margin-right: 20px;
            background-color:
                #70ad47;
            color: white;
            text-align: center;
        }

        #name,
        #major,
        #dob {
            padding: 7px 0 7px 0;
        }

        #major,
        #dob {
            width: 40%;
            text-align: center;
        }

        .name,
        .major,
        .dob,
        .gender,
        .image,
        .address {
            display: flex;
        }

        .name,
        .major,
        .dob,
        .gender,
        .address {
            align-items: baseline;
        }

        .img label {
            max-height: 1rem;
        }


        #fname-validation,
        #d-validation,
        #bd-validation,
        #bd-invalid {
            color: red;
        }

        #address {
            vertical-align: top;
        }

        button {
            font-family: inherit;
            font-size: 1rem;
            color: white;
            width: 30%;
            padding: 10px;
            background-color: #70ad47;
            border-radius: 6px;
            margin: 5% auto 0 auto;
            cursor: pointer;
        }

        .btn {
            text-align: center;
        }

        .star {
            color: red;
        }
    </style>
</head>

<body>
    <div class="outer-box">
        <div class="name">
            <label for="name" class="label">Họ tên<span class="star">*</span></label>
            <?php
            echo "<p>" . $_POST['name'] . "</p>";
            ?>
        </div>
        <div class="gender">
            <label for="gender" class="label">Giới tính<span class="star">*</span></label>
            <?php
            echo "<p>" . $_POST['gender'] . "</p>";
            ?>

        </div>
        <div class="major">
            <label for="major" class="label">Phân khoa<span class="star">*</span></label>
            <?php
            echo "<p>" . $_POST['major'] . "</p>";
            ?>
        </div>
        <div class="dob">
            <label for="dob" class="label">Ngày sinh<span class="star">*</span></label>
            <?php
            echo "<p>" . $_POST['dob'] . "</p>";
            ?>
        </div>
        <div class="address">
            <label for="address" class="label">Địa chỉ</label>
            <?php
            echo "<p>" . $_POST['address'] . "</p>";
            ?>
        </div>
        <div class="inputImage">
            <label for="inputImage" class="label">Hình ảnh</label>
            <?php
            $temp_path = $_FILES["img"]["tmp_name"];
            $image_path = 'uploads/' . $_FILES["img"]["name"];
            if (move_uploaded_file($temp_path, $image_path)) {
                echo "<img src=\"$imagePath\" alt=\"Post Image\" width=\"150\">";
            }
            ?>
        </div>
    </div>
</body>

</html>
