# Project Report: DeepSign Vision - Road Sign Damage Detection

## 1. Why I built this
I've noticed that a lot of road signs in my area are either super faded from the sun or have trees growing over them. It's actually kind of dangerous if you don't know the roads well. I found out that cities usually just have people drive around to find these, which seems really slow. So, I wanted to see if I could build something that detects these signs and also tells you how beat up they are automatically.

## 2. How it works
I built a full-stack project called **DeepSign Vision**. Basically, you upload a picture, and the system does two things:
1. **Finds the sign**: It looks for any road signs in the image.
2. **Checks the damage**: It gives the sign a "Severity Score" from 0 to 100.
    - **0-30 (LOW)**: The sign is fine, maybe a tiny scratch.
    - **31-70 (MEDIUM)**: It's getting hard to read (faded or blurry). I labeled these for review.
    - **71-100 (HIGH)**: This is the "fix it now" category. The sign is basically useless.

## 3. The Tech Stack (What I used)
- **Frontend**: I used **React 19** with **Vite**. I wanted it to look really clean, so I used **Framer Motion** for some smooth animations and a "glassmorphism" look.
- **Backend**: I went with **Node.js (Express)**. I like it because it handles the file uploads easily and can run my Python scripts in the background.
- **AI/ML**: This was the hardest part. I used **YOLOv8** for the fast detection because it's the industry standard right now. But I also added a **ResNet50** classifier (using PyTorch) as a second check to make sure the labels were actually correct.

## 4. Measuring the Damage
I didn't just want a "yes/no" for damage, so I wrote some custom logic to look at:
- **Blur**: I used something called Laplacian variance. If the number is low, the sign is blurry.
- **Color**: I checked the HSV values to see if the sun had faded the paint.
- **Physical Damage**: I used Canny edge detection to see if the sign was cracked or bent.
- **Blocked Signs**: I calculated the ratio of sign-color to the whole area to see if leaves or stickers were covering it.

## 5. Things I struggled with
- **Windows vs Linux issues**: I had this weird bug where my model wouldn't load because of path issues. I had to manually override `pathlib.PosixPath` to `WindowsPath` just to get it working on my laptop. Not the prettiest fix, but it worked!
- **Double Detections**: At first, the AI was showing two boxes for the same sign. I had to write a "Multi-Sign Conflict Resolution" script to check if two boxes were too close together and just keep the better one.
- **React Canvas**: Mapping the box coordinates from the Python script to the React UI was a nightmare because of image scaling. I spent way too long on the math for that.

## 6. What's next?
I'd love to add GPS tracking so you can actually see the damaged signs on a map. Also, I want to pack the whole thing into Docker so it's easier for other people to run without having to install all the Python dependencies.

---
*Note: You can check the training logs for all the boring math stuff like mAP and Precision scores!*

