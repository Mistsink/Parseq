import os
import random as rnd
from typing import List, Tuple

from PIL import Image, ImageFilter, ImageStat

from . import computer_text_generator, background_generator, distorsion_generator
from .util import (
    get_concat_h,
    get_concat_h_left,
    get_concat_v,
    get_concat_v_center,
    mask_to_bboxes,
    make_filename_valid,
)


class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
        Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(
        cls,
        index: int,
        text: str,
        sentences: List[List[Tuple[str, str, str]]],
        out_dir: str,
        size: int,
        extension: str,
        skewing_angle: int,
        random_skew: bool,
        blur: int,
        random_blur: bool,
        background_type: int,
        distorsion_type: int,
        distorsion_orientation: int,
        name_format: int,
        width: int,
        alignment: int,
        text_color: str,
        orientation: int,
        space_width: int,
        character_spacing: int,
        margins: int,
        fit: bool,
        output_mask: bool,
        word_split: bool,
        image_dir: str,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
        image_mode: str = "RGB",
        output_bboxes: int = 0,
    ) -> Image:
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        image, mask = FakeTextDataGenerator.sentences_generate(
            sentences,
            text_color,
            size,
            orientation,
            space_width,
            character_spacing,
            fit,
            word_split,
            stroke_width,
            stroke_fill,
        )
        if image is None:
            if output_mask == 1:
                return None, None
            return None

        if random_skew:
            random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distortion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img  # Mind = blown
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
            distorted_img, distorted_mask = distorsion_generator.sin(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img, distorted_mask = distorsion_generator.cos(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img, distorted_mask = distorsion_generator.random(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # # Horizontal text
        # if orientation == 0:
        #     new_width = int(
        #         distorted_img.size[0]
        #         * (float(size - vertical_margin) / float(distorted_img.size[1]))
        #     )
        #     resized_img = distorted_img.resize(
        #         (new_width, size - vertical_margin), Image.Resampling.LANCZOS
        #     )
        #     resized_mask = distorted_mask.resize(
        #         (new_width, size - vertical_margin), Image.Resampling.NEAREST
        #     )
        #     background_width = width if width > 0 else new_width + horizontal_margin
        #     background_height = size
        # # Vertical text
        # elif orientation == 1:
        #     new_height = int(
        #         float(distorted_img.size[1])
        #         * (float(size - horizontal_margin) / float(distorted_img.size[0]))
        #     )
        #     resized_img = distorted_img.resize(
        #         (size - horizontal_margin, new_height), Image.Resampling.LANCZOS
        #     )
        #     resized_mask = distorted_mask.resize(
        #         (size - horizontal_margin, new_height), Image.Resampling.NEAREST
        #     )
        #     background_width = size
        #     background_height = new_height + vertical_margin
        # else:
        #     raise ValueError("Invalid orientation")
        resized_img, resized_mask = distorted_img, distorted_mask
        background_width = resized_img.width + horizontal_margin
        background_height = resized_img.height + vertical_margin

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background_img = background_generator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background_img = background_generator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background_img = background_generator.quasicrystal(
                background_height, background_width
            )
        else:
            background_img = background_generator.image(
                background_height, background_width, image_dir
            )
        background_width, background_height = background_img.size
        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )

        ##############################################################
        # Comparing average pixel value of text and background image #
        ##############################################################
        try:
            resized_img_st = ImageStat.Stat(resized_img, resized_mask.split()[2])
            background_img_st = ImageStat.Stat(background_img)

            resized_img_px_mean = sum(resized_img_st.mean[:2]) / 3
            background_img_px_mean = sum(background_img_st.mean) / 3

            if abs(resized_img_px_mean - background_img_px_mean) < 15:
                # print("value of mean pixel is too similar. Ignore this image")

                # print("resized_img_st \n {}".format(resized_img_st.mean))
                # print("background_img_st \n {}".format(background_img_st.mean))

                if output_mask == 1:
                    return None, None
                return None
        except Exception as err:
            if output_mask == 1:
                return None, None
            return None

        #############################
        # Place text with alignment #
        #############################

        new_text_width, new_text_height = resized_img.size
        margin_top = (background_height - new_text_height) // 2

        if alignment == 0 or width == -1:
            background_img.paste(resized_img, (margin_left, margin_top), resized_img)
            background_mask.paste(resized_mask, (margin_left, margin_top))
        elif alignment == 1:
            background_img.paste(
                resized_img,
                (int(background_width / 2 - new_text_width / 2), margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (int(background_width / 2 - new_text_width / 2), margin_top),
            )
        else:
            background_img.paste(
                resized_img,
                (background_width - new_text_width - margin_right, margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (background_width - new_text_width - margin_right, margin_top),
            )

        ############################################
        # Change image mode (RGB, grayscale, etc.) #
        ############################################

        background_img = background_img.convert(image_mode)
        background_mask = background_mask.convert(image_mode)

        #######################
        # Apply gaussian blur #
        #######################

        gaussian_filter = ImageFilter.GaussianBlur(
            radius=blur if not random_blur else rnd.random() * blur
        )
        final_image = background_img.filter(gaussian_filter)
        final_mask = background_mask.filter(gaussian_filter)

        #####################################
        # Generate name for resulting image #
        #####################################
        # We remove spaces if space_width == 0
        # if space_width == 0:
        #     text = text.replace(" ", "")
        # if name_format == 0:
        #     name = "{}_{}".format(text, str(index))
        # elif name_format == 1:
        #     name = "{}_{}".format(str(index), text)
        # elif name_format == 2:
        #     name = str(index)
        # else:
        #     print("{} is not a valid name format. Using default.".format(name_format))
        #     name = "{}_{}".format(text, str(index))
        name = str(index)

        name = make_filename_valid(name, allow_unicode=True)
        image_name = "{}.{}".format(name, extension)
        mask_name = "{}_mask.png".format(name)
        box_name = "{}_boxes.txt".format(name)
        tess_box_name = "{}.box".format(name)

        # Save the image
        if out_dir is not None:
            final_image.save(os.path.join(out_dir, image_name))
            if output_mask == 1:
                final_mask.save(os.path.join(out_dir, mask_name))
            if output_bboxes == 1:
                bboxes = mask_to_bboxes(final_mask)
                with open(os.path.join(out_dir, box_name), "w") as f:
                    for bbox in bboxes:
                        f.write(" ".join([str(v) for v in bbox]) + "\n")
            if output_bboxes == 2:
                bboxes = mask_to_bboxes(final_mask, tess=True)
                with open(os.path.join(out_dir, tess_box_name), "w") as f:
                    for bbox, char in zip(bboxes, text):
                        f.write(
                            " ".join([char] + [str(v) for v in bbox] + ["0"]) + "\n"
                        )
        else:
            if output_mask == 1:
                return final_image, final_mask
            return final_image

    @staticmethod
    def sentences_generate(
        sentences: List[List[Tuple[str, str, str]]],
        text_color,
        size,
        orientation,
        space_width,
        character_spacing,
        fit,
        word_split,
        stroke_width,
        stroke_fill,
    ):
        whole_img, whole_mask = None, None
        for sentence_info in sentences:
            sentence_img, sentence_mask = None, None

            for text, lang, font in sentence_info:
                image, mask = computer_text_generator.generate(
                    text,
                    font,
                    text_color,
                    size,
                    orientation,
                    space_width,
                    character_spacing,
                    fit,
                    word_split,
                    stroke_width,
                    stroke_fill,
                )
                # bg = background_generator.plain_white(image.height, image.width)
                # bg.paste(image, (0, 0), image)
                # bg.save("data/tmp/text.png")
                if sentence_img is None or sentence_mask is None:
                    sentence_img = image
                    sentence_mask = mask
                else:
                    if orientation == 0:  # Horizontal
                        # Concatenate horizontally
                        sentence_img = get_concat_h(sentence_img, image)
                        sentence_mask = get_concat_h(sentence_mask, mask)
                    elif orientation == 1:  # Vertical
                        # Concatenate vertically
                        sentence_img = get_concat_v(sentence_img, image)
                        sentence_mask = get_concat_v(sentence_mask, mask)

            # bg = background_generator.plain_white(
            #     sentence_img.height, sentence_img.width
            # )
            # bg.paste(sentence_img, (0, 0), sentence_img)
            # bg.save("data/tmp/text.png")

            # Combine the sentence images and masks into the whole image and mask
            if whole_img is None or whole_mask is None:
                whole_img = sentence_img
                whole_mask = sentence_mask
            else:
                if orientation == 0:  # Horizontal, append to the left
                    whole_img = get_concat_v_center(whole_img, sentence_img)
                    whole_mask = get_concat_v_center(whole_mask, sentence_mask)
                elif orientation == 1:  # Vertical, center horizontally
                    whole_img = get_concat_h_left(whole_img, sentence_img)
                    whole_mask = get_concat_h_left(whole_mask, sentence_mask)

            # bg = background_generator.plain_white(whole_img.height, whole_img.width)
            # bg.paste(whole_img, (0, 0), whole_img)
            # bg.save("data/tmp/text.png")
            pass
        return whole_img, whole_mask