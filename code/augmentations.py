import math
import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import cv2
from wand import image

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='RGB')            
        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask)

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.BILINEAR)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        # assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask
        
class MagnifyData(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img, mask):
        ratio = random.uniform(0.6,1)
        w, h = img.size
        tw = ratio * w
        th = ratio * h
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        return img.resize((self.size, self.size), Image.BILINEAR), mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask
        
class RandomHorizontallyFlipData(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask
        return img, mask
        
class RandomVerticallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask

class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        # assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.BILINEAR)

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask): # longer side of image is scaled to defined size.
        warning_size = 5

        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            pass
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)

            if oh < warning_size:
                print('warning: resized image height is less than %d'%warning_size)

            img = img.resize((ow, oh), Image.BILINEAR)
        else:
            oh = self.size
            ow = int(self.size * w / h)

            if ow < warning_size:
                print('warning: resized image width is less than %d'%warning_size)

            img = img.resize((ow, oh), Image.BILINEAR)
        
        w, h = mask.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            pass
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)

            if oh < warning_size:
                print('warning: resized template height is less than %d'%warning_size)

            mask = mask.resize((ow, oh), Image.BILINEAR)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            
            if ow < warning_size:
                print('warning: resized template width is less than %d'%warning_size)
            
            mask = mask.resize((ow, oh), Image.BILINEAR)

        return img, mask

class CenterPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask): # longer side of image is scaled to defined size.
        w, h = img.size

        assert self.size[0] >= h
        assert self.size[1] >= w
        w_pad = self.size[1]-w
        w_pad_left = int(w_pad/2)
        w_pad_right = w_pad - w_pad_left
        h_pad = self.size[0]-h
        h_pad_up = int(h_pad/2)
        h_pad_bottom = h_pad - h_pad_up
        padding = (w_pad_left, h_pad_up, w_pad_right, h_pad_bottom)

        img = ImageOps.expand(img, border=padding, fill=0)        

        w, h = mask.size

        assert self.size[0] >= h
        assert self.size[1] >= w
        w_pad = self.size[1]-w
        w_pad_left = int(w_pad/2)
        w_pad_right = w_pad - w_pad_left
        h_pad = self.size[0]-h
        h_pad_up = int(h_pad/2)
        h_pad_bottom = h_pad - h_pad_up
        padding = (w_pad_left, h_pad_up, w_pad_right, h_pad_bottom)

        mask = ImageOps.expand(mask, border=padding, fill=0) 

        return img, mask

class ReflectPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask): # longer side of image is scaled to defined size.
        w, h = img.size

        assert self.size[0] >= h
        assert self.size[1] >= w
        img = np.array(img)
        margin = 5
        if self.size[1] > w:
            w_pad = (self.size[1]-w)/2.
            pad_times = int(w_pad/margin)
            for i in range(pad_times):
                img = cv2.copyMakeBorder(img,margin,0,margin,0,cv2.BORDER_REFLECT)
        if self.size[0] > h:
            h_pad = (self.size[1]-h)/2.
            pad_times = int(h_pad/margin)
            for i in range(pad_times):
                img = cv2.copyMakeBorder(img,0,margin,0,margin,cv2.BORDER_REFLECT)
        img = Image.fromarray(img)
        
        w, h = mask.size

        assert self.size[0] >= h
        assert self.size[1] >= w
        w_pad = self.size[1]-w
        w_pad_left = int(w_pad/2)
        w_pad_right = w_pad - w_pad_left
        h_pad = self.size[0]-h
        h_pad_up = int(h_pad/2)
        h_pad_bottom = h_pad - h_pad_up
        padding = (w_pad_left, h_pad_up, w_pad_right, h_pad_bottom)

        mask = ImageOps.expand(mask, border=padding, fill=0)

        return img, mask

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        # assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size), Image.BILINEAR)
        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))
        
class RandomCropData(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        # assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.5, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask
        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.BILINEAR)
        
class RandomRotateData(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask
        
class FixedRotate(object):
    def __call__(self, img, mask):
        idx = random.randint(0,7)
        rotate_degree = (idx - 3) * 45
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.BILINEAR)

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.BILINEAR)

        return self.crop(*self.scale(img, mask))


class RandomRotateRefPadding(object):
    def __init__(self, degree):
        self.degree = degree
        
    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        rad_degree = math.radians(abs(rotate_degree))
        margin = 5
        h, w = img.size[0], img.size[1]
        h_padded = h + w * math.sin(2 * rad_degree)
        w_padded = w + h * math.sin(2 * rad_degree)
        pad_times = max(int((h_padded-h)/margin/2.), int((w_padded-w)/margin/2.))
        img = np.array(img)
        for i in range(pad_times):
            img = cv2.copyMakeBorder(img,margin,margin,margin,margin,cv2.BORDER_REFLECT)
        img = Image.fromarray(img)

        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.BILINEAR)

class RandomWarpData(object):
    def __call__(self, img, mask):
        img = np.asarray(img)
        # for gtsrb
        #a = random.uniform(0.8,1.0)
        #e = random.uniform(0.8,1.0)
        #b = random.uniform(-0.2,0.2)
        #d = random.uniform(-0.2,0.2)
        #c = random.uniform(-1,1)
        #f = random.uniform(-1,1)
        
        #for brand
        a = random.uniform(0.8,1.0)
        e = random.uniform(0.8,1.0)
        b = random.uniform(-0.2,0.2)
        d = random.uniform(-0.2,0.2)
        c = random.uniform(-1,1)
        f = random.uniform(-1,1)       
        
        theta = np.array([[a, b, c], [ d, e, f]])
        img = cv2.warpAffine(img, theta, (img.shape[1], img.shape[0]))
        img = Image.fromarray(img)
        return img, mask
        
class RandomEnhanceData(object):
    def __call__(self, img, mask):
        brightness = random.uniform(0.2,2)
        color = random.uniform(0.1,2)
        contrast = random.uniform(0.1,2)
        sharpness = random.uniform(0.1,2)
        img = ImageEnhance.Brightness(img)
        img = img.enhance(brightness)
        img = ImageEnhance.Color(img)
        img = img.enhance(color)
        img = ImageEnhance.Contrast(img)
        img = img.enhance(contrast)
        img = ImageEnhance.Sharpness(img)
        img = img.enhance(sharpness)        
        return img, mask
            
class RandomDownData(object):
    def __call__(self, img, mask):
        w, h = img.size
        new_size = random.uniform(10, min(w, h))
        img.thumbnail([new_size, new_size])
        img = img.transform((w, h), Image.EXTENT, (0, 0, new_size, new_size))
        return img, mask
        
class OverExposeData(object):
    def __call__(self, img, mask):
         img = np.array(img)
         img[img[:,:,0]>230,1] = 240
         img[img[:,:,0]>230,2] = 240
         img = Image.fromarray(img)
         return img, mask
         
class GaussianBlurData(object):
    def __call__(self, img, mask):
        img = np.array(img)
        s = random.randint(5,20)
        img = cv2.GaussianBlur(img, ksize=(2*s+1,2*s+1), sigmaX=0, sigmaY=0)
        img = Image.fromarray(img)
        return img, mask
        
class MotionBlurData(object):
    def __call__(self, img, mask):
        degree = random.randint(2,20)
        angle = random.uniform(2,90)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        
        img = np.array(img)
        img = cv2.filter2D(img, -1, motion_blur_kernel)
        #cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)
        img = np.array(img, dtype=np.uint8)
        img = Image.fromarray(img)
        return img, mask
        
class RandomStainData(object):
    def __call__(self, img, mask):
         img = np.array(img)
         s = img.shape[0]
         x_c = int(random.uniform(0.4, 0.6) * s)
         y_c = int(random.uniform(0.4, 0.6) * s)
         x = int(random.uniform(0.05, 0.1) * s)
         y = int(random.uniform(0.05, 0.1) * s)
         img[y_c-y:y_c+y, x_c-x:x_c+x,:] = 255
         img = Image.fromarray(img)
         return img, mask
         
class BlurData(object):
    def __call__(self, img, mask):
        box_para = random.uniform(1, 2)
        gauss_para = random.uniform(1, 2)
        #img = img.filter(ImageFilter.BLUR)
        if random.random() < 0.5:
            img = img.filter(ImageFilter.BoxBlur(box_para))
        else:
            img = img.filter(ImageFilter.GaussianBlur(gauss_para))
        return img, mask
        
class EnhanceTemp(object):
    def __call__(self, img, mask):
        brightness = 1
        color = 3
        contrast = 1
        sharpness = 1
        mask = ImageEnhance.Brightness(mask)
        mask = mask.enhance(brightness)
        mask = ImageEnhance.Color(mask)
        mask = mask.enhance(color)
        mask = ImageEnhance.Contrast(mask)
        mask = mask.enhance(contrast)
        mask = ImageEnhance.Sharpness(mask)
        mask = mask.enhance(sharpness)        
        return img, mask
        
class PaddingData(object):
    def __call__(self, img, mask):
        pad_times = 1
        margin = 5
        img = np.array(img)
        for i in range(pad_times):
            img = cv2.copyMakeBorder(img, margin, margin, margin, margin, cv2.BORDER_CONSTANT,value=[255,255,255])
        img = Image.fromarray(img)
        return img, mask
        
class PixelEnhanceData(object):
    def __call__(self, img, mask):
        number = random.randint(0,100)
        channel = random.randint(0,2)
        img = np.array(img)
        img_sum = np.sum(img, 2)
        #img[img_sum<10,:] += number
        img[:,:,channel] += number
        img = Image.fromarray(img)
        return img, mask
        
class RandomColorData(object):
    def __call__(self, img, mask):     
        img = np.array(img)
        color = np.random.randint(0,150,img.shape)
        img = img + color
        img[img>255] = 255
        img = np.array(img, dtype=np.uint8)
        img = Image.fromarray(img)
        return img, mask
        
class InverseColorData(object):
    def __call__(self, img, mask):     
        img = np.array(img)
        if random.random() < 0.5:
            img = 255 - img
        img = Image.fromarray(img)
        return img, mask
        
class DistortData(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            angle = random.randint(1, 10) * 10
            if random.random() < 0.5:
                img = img.rotate(180, Image.BILINEAR)
                img = np.array(img)
                with image.Image.from_array(img) as img:
                    img.distort('arc', (angle,180))
                    img = np.array(img)
            else:
                img = np.array(img)
                with image.Image.from_array(img) as img:
                    img.distort('arc', (angle,))
                    img = np.array(img)
            img = Image.fromarray(img)
        return img, mask
        
class RandomBlockData(object):
    def __call__(self, img, mask):
        img = np.array(img)
        h = img.shape[0]
        w = img.shape[1]
        #y1 = random.randint(0, h)
        #y2 = random.randint(y1, int(h/2)*(int(h/2)>y1) + h*(int(h/2)<=y1))
        #x1 = random.randint(0, w)
        #x2 = random.randint(x1, int(w/2)*(int(w/2)>x1) + w*(int(w/2)<=x1))
        
        y1 = random.randint(0, int(h/2))
        y2 = random.randint(int(h/2), h)
        x1 = random.randint(0, int(w/2))
        x2 = random.randint(int(w/2), w)
        
        block = np.random.randint(0,255,(y2-y1,x2-x1,img.shape[2]))
        img[y1:y2, x1:x2, :] = block
        img = Image.fromarray(img)
        return img, mask
        
        
        